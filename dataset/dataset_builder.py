import json
import traceback
import multiprocessing as mp
import os
import time
from pathlib import Path
import numpy as np
from ..config import GeneratorConfig
from ..exceptions import GenerationError
from ..validators.geometry_validator import ValidationError
from ..builder.ast_builder import ASTBuilder
from ..executor.executor import Executor
from ..validators.geometry_validator import GeometryValidator
from ..exporters.step_exporter import StepExporter
from ..exporters.stl_exporter import StlExporter
from ..exporters.pointcloud_exporter import PointCloudExporter
from ..exporters.mesh_exporter import MeshExporter
from ..exporters.render_exporter import RenderExporter
from ..metadata.metadata_manager import Metadata
from ..syntax_tree.nodes import ASTNode
from ..code_generator import generate_code
import logging

logger = logging.getLogger(__name__)

class DatasetBuilder:
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.executor = Executor()
        self.validator = GeometryValidator(config)

    @property
    def _programs_only(self) -> bool:
        """Режим «только программы»: в save_formats нет ни одного геометрического
        экспорта, значит на сэмпл пишется ровно один (или два) текстовых файла."""
        return set(self.config.save_formats) <= {"program_py", "program_txt"}

    def _sample_base_dir(self, complexity: str) -> Path:
        base = self.config.output_dir
        if self.config.split_by_complexity:
            base = base / complexity
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _sample_timeout(self, index: int) -> float:
        """Таймаут сэмпла по его сложности: лёгкие уровни не должны ждать
        150s, зависший воркер прибивается раньше (см. config.sample_timeouts)."""
        complexity = self.config.complexity_levels[index % len(self.config.complexity_levels)]
        return float(self.config.sample_timeouts.get(complexity, self.config.sample_timeout))

    def generate_sample(self, index: int, seed: int) -> bool:
        self.executor.clear()
        try:
            complexity = self.config.complexity_levels[index % len(self.config.complexity_levels)]
            builder = ASTBuilder(self.config, seed=seed, complexity=complexity)
            ast = builder.build()
            shape = self.executor.execute(ast)
            self.validator.validate(shape, is_root=True)
            base_dir = self._sample_base_dir(complexity)
            if self._programs_only:
                # режим по умолчанию: только исполняемые программы, разбитые
                # по папкам easy/medium/hard/expert — без подпапок и экспортов
                code = generate_code(ast)
                sample_file = base_dir / f"sample_{index:05d}.py"
                sample_file.write_text(code, encoding="utf-8")
                if "program_txt" in self.config.save_formats:
                    (base_dir / f"sample_{index:05d}.txt").write_text(code, encoding="utf-8")
                logger.info(f"Sample {index} generated successfully.")
                return True
            sample_dir = base_dir / f"sample_{index:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            # метаданные считаем ДО validate/export: BRepMesh_IncrementalMesh
            # (в fast_bbox и tessellate) мутирует shape, после чего BoundingBox()
            # возвращает другие значения
            metadata = Metadata.from_ast(ast, complexity, seed, self.config, shape=shape)
            self._export(ast, shape, sample_dir)
            self._save_program(ast, sample_dir)
            # metadata.json/ast.json пишутся всегда в полном режиме: они дёшевы
            # и описывают сэмпл (сложность, seed, состав операций)
            with open(sample_dir / "metadata.json", "w") as f:
                f.write(metadata.model_dump_json(indent=2))
            logger.info(f"Sample {index} generated successfully.")
            return True
        except Exception as e:
            complexity = self.config.complexity_levels[index % len(self.config.complexity_levels)]
            if isinstance(e, (GenerationError, ValidationError)):
                # известные классы отказов сэмпла — одна строка (без толстого
                # traceback): причина уже зашита в сообщение, сэмпл уйдёт на retry
                logger.error(f"Failed to generate sample {index} ({complexity}): "
                             f"{type(e).__name__}: {e}")
            else:
                # неизвестный класс — полный трейсбек для диагностики
                logger.error(f"Failed to generate sample {index} ({complexity}): {e}")
                traceback.print_exc()
            sample_dir = self._sample_base_dir(complexity) / f"sample_{index:05d}"
            if not self._programs_only and sample_dir.exists():
                import shutil
                shutil.rmtree(sample_dir)
            return False

    def _export(self, ast: ASTNode, shape, sample_dir: Path):
        formats = self.config.save_formats
        if "step" in formats:
            StepExporter.export(shape, sample_dir / "model.step")
        if "stl" in formats:
            StlExporter.export(shape, sample_dir / "model.stl")
        if "pointcloud" in formats:
            PointCloudExporter.export(shape, sample_dir / "pointcloud.npy", self.config.pointcloud_samples)
        if "mesh" in formats:
            MeshExporter.export(shape, sample_dir / "mesh.obj")
        if "render" in formats:
            RenderExporter.export(shape, sample_dir / "render.png", self.config.render_resolution)
        # ast.json пишется всегда в полном режиме (как и metadata.json): он нужен
        # для обучения/воспроизведения и дёшев; формат-элемент оставлен для явной
        # совместимости с флагом --export-formats
        with open(sample_dir / "ast.json", "w") as f:
            f.write(ast.to_json())

    def _save_program(self, ast: ASTNode, sample_dir: Path):
        code = generate_code(ast)
        if "program_py" in self.config.save_formats:
            with open(sample_dir / "program.py", "w") as f:
                f.write(code)
        if "program_txt" in self.config.save_formats:
            with open(sample_dir / "program.txt", "w") as f:
                f.write(code)

    def generate_dataset(self, num_samples: int, parallel: bool = True,
                         start_sample: int = 0):
        """Генерирует датасет, защищая каждый сэмпл жёстким таймаутом.

        OCCT-операции (булевы, массивы) могут недетерминированно зависать на
        отдельных семплях. Каждый сэмпл выполняется в отдельном воркере
        multiprocessing.Pool; если сэмпл не завершился за sample_timeout секунд,
        пул принудительно завершается и сэмпл перезапускается со следующим seed.
        Зависшие сэмплы изолированы: независшим воркерам даётся grace-интервал
        закончить работу до terminate(), и каждый перезапуск тратит retry
        (цикл гарантированно сходится). Внутри билдера сборка ограничена
        wall-clock лимитом, поэтому превышение sample_timeout означает зависшую
        OCCT-операцию. Ни один «плохой» seed не может заблокировать генерацию.

        `start_sample` — индекс первого сэмпла (по умолчанию 0): с него начинается
        генерация. Семена считаются по абсолютному диапазону
        `SeedSequence(global_seed).generate_state(start_sample + num_samples)` и
        берётся срез `[start_sample : start_sample + num_samples]`, поэтому
        `seed(i) = states[i]` не зависит от размера чанка: прогон 30 сэмплов за
        раз и два прогона по 15 (0..14 + 15..29) при том же `--seed` дают
        идентичные программы. Работает во всех режимах вывода (только программы,
        плоский `--flat`, полный с экспортами): сложность, имена файлов
        `sample_{index:05d}*` и таймауты привязаны к абсолютному индексу.

        Индексы обрабатываются **окнами** ~`4×workers`: зависший или падающий
        сэмпл переделывает только своё окно (перезапуск/повтор по прежней схеме),
        а не переотправляет весь оставшийся хвост. Каждое окно — свой
        `multiprocessing.Pool`. Семантика seed-ретраев и `max_seed_retries`
        сохраняется; на выходе датасет идентичен «окну = всему хвосту».
        """
        if num_samples <= 0:
            logger.warning("num_samples=%d: nothing to generate", num_samples)
            return
        if start_sample < 0:
            raise ValueError(f"start_sample must be >= 0, got {start_sample}")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        total = start_sample + num_samples
        base_seeds = np.random.SeedSequence(self.config.global_seed).generate_state(total)
        indices = list(range(start_sample, total))
        workers = 1
        if parallel:
            # 8 параллельных OCCT-процессов достаточно, чтобы насытить CPU;
            # массовый spawn десятков процессов на Windows может сбоить (WinError 87)
            workers = max(1, min(os.cpu_count() or 1, 8, num_samples))
        window_size = max(4, 4 * workers)
        retries_left = {i: self.config.max_seed_retries for i in indices}
        pending: dict = {i: int(base_seeds[i]) for i in indices}
        logger.info("Generating %d samples starting from index %d "
                    "(global seed %d)", num_samples, start_sample,
                    self.config.global_seed)

        while pending:
            # окно не длиннее 4×workers: крупная авария переделается локально
            _items = list(pending.items())
            window: dict = dict(_items[:window_size])
            rest: dict = dict(_items[window_size:])
            pool = None
            for _ in range(3):
                try:
                    pool = mp.Pool(processes=min(workers, len(window)))
                    break
                except OSError:
                    time.sleep(1.0)
            if pool is None:
                raise RuntimeError("failed to create multiprocessing pool")
            futures = {i: pool.apply_async(self.generate_sample, (i, seed))
                       for i, seed in window.items()}
            pool.close()

            next_pending: dict = {}
            settled = set()
            hung = None

            def settle(idx: int, ok: bool):
                """Учитывает результат сэмпла: удаляет либо отправляет на retry."""
                retries_left[idx] -= 1
                if ok:
                    return
                if retries_left[idx] <= 0:
                    logger.error(f"Sample {idx} failed after "
                                 f"{self.config.max_seed_retries} seed retries")
                else:
                    prev = window[idx]
                    next_pending[idx] = prev + 1

            for i in list(futures):
                if hung is not None:
                    break
                try:
                    ok = bool(futures[i].get(timeout=self._sample_timeout(i)))
                except mp.TimeoutError:
                    hung = i
                    break
                except Exception as exc:
                    logger.error(f"Sample {i} raised an exception: {exc}")
                    ok = False
                settled.add(i)
                settle(i, ok)

            if hung is not None:
                logger.error(f"Sample {hung} timed out after "
                             f"{self._sample_timeout(hung)}s "
                             f"(OCCT op hung) — restarting with next seed")
                # доводим независших товарищей: пока они ещё выполнялись, даём
                # им ещё один интервал (grace) закончиться, чтобы terminate()
                # не выбрасывал их готовую работу. Grace = максимум таймаута
                # среди оставшихся, а не фиксированные 150s.
                remaining = [j for j in futures if j != hung and j not in settled]
                grace_deadline = time.monotonic() + (max(self._sample_timeout(j) for j in remaining)
                                                     if remaining else 0.0)
                for j in list(futures):
                    if j == hung or j in settled:
                        continue
                    rem = grace_deadline - time.monotonic()
                    if rem <= 0:
                        break
                    try:
                        ok = bool(futures[j].get(timeout=rem))
                    except mp.TimeoutError:
                        continue
                    except Exception as exc:
                        logger.error(f"Sample {j} raised an exception: {exc}")
                        ok = False
                    settled.add(j)
                    settle(j, ok)
                retries_left[hung] -= 1
                if retries_left[hung] > 0:
                    next_pending[hung] = window[hung] + 1
            # сэмплы, чья работа оборвана terminate() до завершения, запускаем
            # заново с тем же seed; такой перезапуск тоже тратит retry, чтобы
            # цикл всегда сходился (нельзя вечно перезапускать «медленных»)
            for j in list(futures):
                if j in settled:
                    continue
                retries_left[j] -= 1
                if retries_left[j] > 0:
                    next_pending.setdefault(j, window[j])
                else:
                    logger.error(f"Sample {j} repeatedly killed by timeouts after "
                                 f"{self.config.max_seed_retries} attempts")

            pool.terminate()
            pool.join()
            pending = next_pending
            pending.update(rest)
        logger.info("Dataset generation finished.")
