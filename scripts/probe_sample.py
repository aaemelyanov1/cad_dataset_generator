"""Одиночный сэмпл для harness-диагностики: строит AST, исполняет холодным
Executor'ом, валидирует корень и печатает ОДНУ JSON-строку результата.

Запускается debug_generate.py как отдельный процесс (изоляция OCCT-крашей,
как в боевом multiprocessing-воркере). exit code < 0 (например
-1073741819 / STATUS_ACCESS_VIOLATION) — жёсткий крах процесса.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cad_dataset_generator.config import GeneratorConfig
from cad_dataset_generator.builder.ast_builder import ASTBuilder
from cad_dataset_generator.executor.executor import Executor
from cad_dataset_generator.validators.geometry_validator import GeometryValidator
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
    if type(exc).__name__ == "ValidationError":
        return "validation"
    if isinstance(exc, GenerationError):
        return "generation"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--complexity", type=str, required=True)
    parser.add_argument("--timeout", type=float, default=150.0)
    args = parser.parse_args()

    config = GeneratorConfig(global_seed=args.seed)
    t0 = time.monotonic()
    try:
        builder = ASTBuilder(config, seed=args.seed, complexity=args.complexity)
        ast = builder.build()
        shape = Executor().execute(ast)
        GeometryValidator(config).validate(shape, is_root=True)
        elapsed = time.monotonic() - t0
        x0, x1, y0, y1, z0, z1 = fast_bbox(shape)
        diag = math.hypot(x1 - x0, y1 - y0, z1 - z0)
        out = {"ok": True, "error_class": None, "detail": "", "elapsed": elapsed,
               "bbox_diag": diag, "volume": float(shape.Volume()),
               "ops": ast.count_operations()}
    except BaseException as exc:
        elapsed = time.monotonic() - t0
        out = {"ok": False, "error_class": _classify(args.complexity, exc, elapsed,
                                                      args.timeout),
               "detail": f"{type(exc).__name__}: {exc}"[:400], "elapsed": elapsed,
               "bbox_diag": None, "volume": None, "ops": None}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()