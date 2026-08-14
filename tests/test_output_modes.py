"""Тесты режимов вывода: по умолчанию только программы в easy/medium/hard/expert,
полный режим с экспортами по запросу, плоский режим --flat."""
import pytest

from cad_dataset_generator.config import GeneratorConfig
from cad_dataset_generator.dataset.dataset_builder import DatasetBuilder


def test_default_writes_only_py_split_by_complexity(tmp_path):
    config = GeneratorConfig(output_dir=tmp_path)
    builder = DatasetBuilder(config)
    builder.generate_dataset(4, parallel=False)
    for i, comp in enumerate(["easy", "medium", "hard", "expert"]):
        p = tmp_path / comp / f"sample_{i:05d}.py"
        assert p.exists(), f"missing {p}"
        code = p.read_text(encoding="utf-8")
        assert "cq.Workplane" in code
    assert (tmp_path / "sample_00000").exists() is False


def test_export_formats_write_subfolders(tmp_path):
    config = GeneratorConfig(output_dir=tmp_path,
                             save_formats=["program_py", "step", "stl"])
    builder = DatasetBuilder(config)
    builder.generate_dataset(1, parallel=False)
    sample_dir = tmp_path / "easy" / "sample_00000"
    assert (sample_dir / "program.py").exists()
    assert (sample_dir / "metadata.json").exists()
    assert (sample_dir / "ast.json").exists()
    assert (sample_dir / "model.step").exists()
    assert (sample_dir / "model.stl").exists()


def test_flat_mode_writes_flat_folder(tmp_path):
    config = GeneratorConfig(output_dir=tmp_path, split_by_complexity=False)
    builder = DatasetBuilder(config)
    builder.generate_dataset(2, parallel=False)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["sample_00000.py", "sample_00001.py"]


def test_program_txt_only_still_single_file_mode(tmp_path):
    config = GeneratorConfig(output_dir=tmp_path,
                             save_formats=["program_py", "program_txt"])
    builder = DatasetBuilder(config)
    builder.generate_dataset(1, parallel=False)
    assert (tmp_path / "easy" / "sample_00000.py").exists()
    assert (tmp_path / "easy" / "sample_00000.txt").exists()
