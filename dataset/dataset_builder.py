import json
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from ..config import GeneratorConfig
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

    def generate_sample(self, index: int, seed: int) -> bool:
        sample_dir = self.config.output_dir / f"sample_{index:05d}"
        self.executor.clear()
        try:
            sample_dir.mkdir(parents=True, exist_ok=True)
            complexity = self.config.complexity_levels[index % len(self.config.complexity_levels)]
            builder = ASTBuilder(self.config, seed=seed, complexity=complexity)
            ast = builder.build()
            shape = self.executor.execute(ast)
            # метаданные считаем ДО validate/export: BRepMesh_IncrementalMesh
            # (в fast_bbox и tessellate) мутирует shape, после чего BoundingBox()
            # возвращает другие значения
            metadata = Metadata.from_ast(ast, complexity, seed, self.config, shape=shape)
            self.validator.validate(shape, is_root=True)
            self._export(ast, shape, sample_dir)
            self._save_program(ast, sample_dir)
            with open(sample_dir / "metadata.json", "w") as f:
                f.write(metadata.model_dump_json(indent=2))
            logger.info(f"Sample {index} generated successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to generate sample {index}: {e}")
            traceback.print_exc()
            if sample_dir.exists():
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
        if "ast_json" in formats:
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

    def generate_dataset(self, num_samples: int, parallel: bool = True):
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        base_seeds = np.random.SeedSequence(self.config.global_seed).generate_state(num_samples)
        if parallel:
            self._generate_parallel(num_samples, base_seeds)
        else:
            for i in range(num_samples):
                base = int(base_seeds[i])
                for retry in range(self.config.max_seed_retries):
                    if self.generate_sample(i, base + retry):
                        break
                else:
                    logger.error(
                        f"Sample {i} failed after {self.config.max_seed_retries} seed retries")

    def _generate_parallel(self, num_samples: int, base_seeds) -> None:
        retries_left = {i: self.config.max_seed_retries for i in range(num_samples)}
        attempts: dict = {}
        with ProcessPoolExecutor() as pool:
            for i in range(num_samples):
                seed = int(base_seeds[i])
                attempts[pool.submit(self.generate_sample, i, seed)] = (i, seed)
            while attempts:
                for future in as_completed(attempts):
                    i, seed = attempts.pop(future)
                    ok = False
                    try:
                        ok = bool(future.result())
                    except Exception as exc:
                        logger.error(f"Sample {i} (seed {seed}) raised an exception: {exc}")
                    if not ok:
                        retries_left[i] -= 1
                        if retries_left[i] > 0:
                            attempts[pool.submit(self.generate_sample, i, seed + 1)] = (i, seed + 1)
                        else:
                            logger.error(
                                f"Sample {i} failed after {self.config.max_seed_retries} seed retries")
                    break
