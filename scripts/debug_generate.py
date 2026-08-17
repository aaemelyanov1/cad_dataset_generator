"""Отладочный harness генерации: проходит N программ тем же путём, что и
боевой запуск (`main.py --num_samples N --seed ... --parallel --flat`), но БЕЗ
seed-retry — каждый ошибки-класс считается один раз на индекс.

Для каждого индекса (сложность по кругу `index % 4`) выполняет:
  ASTBuilder.build()  (внутри — холодная root-валидация, config.verify_root_cold)
  -> свежий Executor().execute(ast)
  -> GeometryValidator.validate(shape, is_root=True)
и собирает: класс ошибки (или OK), тайминги, bbox-диагональ, объём, число операций.

Семена те же, что у боевого запуска: SeedSequence(global_seed).generate_state(n),
срез [start : start+N] -> seed(i)=states[i]. Воркеры = max(1, min(cpu, 8, N)).

Не входит в дефолтный pytest.
Пример: python scripts/debug_generate.py --num_samples 1000 --seed 42 --parallel \
        --out metrics.json
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# репозиторий лежит в папке ровно с именем пакета `cad_dataset_generator`,
# поэтому корень импорта — на два уровня выше (как в main.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from cad_dataset_generator.config import GeneratorConfig
from cad_dataset_generator.builder.ast_builder import ASTBuilder
from cad_dataset_generator.executor.executor import Executor
from cad_dataset_generator.validators.geometry_validator import GeometryValidator, ValidationError
from cad_dataset_generator.exceptions import GenerationError
from cad_dataset_generator.utils.geometry_utils import fast_bbox


def _classify(complexity: str, exc: BaseException, elapsed: float, timeout: float) -> str:
    if elapsed > timeout:
        return "hang"
    msg = str(exc)
    if "split produced" in msg or "split:" in msg:
        return "split"
    if "Solid (got Compound)" in msg or "is not a Solid" in msg:
        return "root_compound"
    if "Null TopoDS_Shape" in msg:
        return "null_shape"
    if isinstance(exc, ValidationError):
        return "validation"
    if isinstance(exc, GenerationError):
        return "generation"
    return "other"


def run_one(args):
    idx, seed, complexity, timeout, cfg = args
    config = GeneratorConfig(**cfg)
    t0 = time.monotonic()
    try:
        builder = ASTBuilder(config, seed=seed, complexity=complexity)
        ast = builder.build()
        shape = Executor().execute(ast)
        validator = GeometryValidator(config)
        validator.validate(shape, is_root=True)
        elapsed = time.monotonic() - t0
        x0, x1, y0, y1, z0, z1 = fast_bbox(shape)
        diag = math.hypot(x1 - x0, y1 - y0, z1 - z0)
        vol = float(shape.Volume())
        return {"idx": idx, "ok": True, "error_class": None, "detail": "",
                "elapsed": elapsed, "bbox_diag": diag, "volume": vol,
                "ops": ast.count_operations(), "complexity": complexity}
    except BaseException as exc:
        elapsed = time.monotonic() - t0
        return {"idx": idx, "ok": False,
                "error_class": _classify(complexity, exc, elapsed, timeout),
                "detail": f"{type(exc).__name__}: {exc}"[:400],
                "elapsed": elapsed, "bbox_diag": None, "volume": None,
                "ops": None, "complexity": complexity}


def main():
    parser = argparse.ArgumentParser(description="Harness генерации 1000 (без seed-retry)")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42, help="Глобальный seed")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--mode", choices=["subprocess", "pool"], default="subprocess",
                        help="subprocess: каждый сэмпл в отдельном процессе "
                             "(класс 'crash' по exit-коду, макс. изоляция OCCT); "
                             "pool: multiprocessing.Pool (как боевой пул)")
    parser.add_argument("--out", type=str, default="metrics.json",
                        help="Путь к выходному JSON-отчёту")
    args = parser.parse_args()

    n, start, total = args.num_samples, args.start, args.start + args.num_samples
    base_seeds = np.random.SeedSequence(args.seed).generate_state(total)
    workers = 1
    if args.parallel:
        workers = max(1, min(os.cpu_count() or 1, 8, n))

    cfg = GeneratorConfig(global_seed=args.seed).__dict__.copy()
    levels = cfg["complexity_levels"]

    tasks = [(i, int(base_seeds[i]), levels[i % len(levels)],
              float(cfg["sample_timeouts"].get(levels[i % len(levels)],
                                               cfg["sample_timeout"])), cfg)
             for i in range(start, total)]

    t_all = time.monotonic()
    probe = Path(__file__).resolve().with_name("probe_sample.py")

    def run_probe(task) -> dict:
        idx, seed, complexity, timeout, _ = task
        cmd = [sys.executable, "-u", str(probe), "--idx", str(idx), "--seed", str(seed),
               "--complexity", complexity, "--timeout", str(timeout)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"idx": idx, "ok": False, "error_class": "hang", "detail": "timeout",
                    "elapsed": timeout, "bbox_diag": None, "volume": None, "ops": None,
                    "complexity": complexity}
        if proc.returncode != 0 or not proc.stdout.strip():
            return {"idx": idx, "ok": False, "error_class": "crash",
                    "detail": f"exit={proc.returncode}",
                    "elapsed": None, "bbox_diag": None, "volume": None, "ops": None,
                    "complexity": complexity}
        data = json.loads(proc.stdout.strip().splitlines()[-1])
        data["idx"] = idx
        data["complexity"] = complexity
        return data

    if args.mode == "subprocess":
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = [r for r in ex.map(run_probe, tasks)]
    else:
        if workers > 1 and len(tasks) > 1:
            with mp.Pool(processes=workers) as pool:
                results = [r for r in pool.imap_unordered(run_one, tasks)]
        else:
            results = [run_one(t) for t in tasks]
    wall = time.monotonic() - t_all

    results.sort(key=lambda r: r["idx"])
    by_class: dict = {}
    by_complexity: dict = {}
    for r in results:
        cls = r["error_class"] or "ok"
        by_class[cls] = by_class.get(cls, 0) + 1
        by_complexity.setdefault(r["complexity"], []).append(r)
    diags = [r["bbox_diag"] for r in results if r["ok"] and r["bbox_diag"]]
    elapsed_ok = [r["elapsed"] for r in results if r["ok"]]

    def p90(xs):
        return float(sorted(xs)[int(len(xs) * 0.9)]) if xs else None

    summary = {
        "num_samples": n, "start": start, "seed": args.seed, "workers": workers,
        "wall_seconds": wall,
        "ok": sum(1 for r in results if r["ok"]),
        "fail": sum(1 for r in results if not r["ok"]),
        "error_classes": dict(sorted(by_class.items(), key=lambda kv: -kv[1])),
        "per_complexity": {
            c: {
                "ok": sum(1 for r in rs if r["ok"]),
                "fail": sum(1 for r in rs if not r["ok"]),
                "mean_elapsed": (sum(r["elapsed"] for r in rs if r["ok"]) /
                                 max(1, sum(1 for r in rs if r["ok"]))),
                "p90_elapsed": p90([r["elapsed"] for r in rs if r["ok"]]),
                "max_elapsed": max((r["elapsed"] for r in rs if r["ok"]), default=None),
            }
            for c, rs in sorted(by_complexity.items())
        },
        "bbox_diag": {
            "min": min(diags) if diags else None,
            "max": max(diags) if diags else None,
            "ratio": (max(diags) / min(diags)) if len(diags) > 1 and min(diags) > 0 else None,
        },
        "elapsed_ok_seconds": {
            "mean": sum(elapsed_ok) / len(elapsed_ok) if elapsed_ok else None,
            "p90": p90(elapsed_ok),
            "max": max(elapsed_ok) if elapsed_ok else None,
        },
    }

    out = Path(args.out)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for r in results:
        if not r["ok"]:
            et = f"{r['elapsed']:.2f}" if r.get("elapsed") is not None else "--"
            print(f"  FAIL idx={r['idx']} {r['complexity']} [{r['error_class']}] "
                  f"{et}s {r['detail']}")


if __name__ == "__main__":
    main()