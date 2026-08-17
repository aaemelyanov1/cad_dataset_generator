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


def test_start_sample_chunk_determinism(tmp_path):
    """Прогон 6 сэмплов одним заходом == два прогона по 3 (0..2 + 3..5):
    тот же --seed и абсолютные индексы дают идентичные программы."""
    cfg = dict(complexity_levels=["easy"], split_by_complexity=False)
    a = tmp_path / "one_go"
    b = tmp_path / "two_goes"
    builder_a = DatasetBuilder(GeneratorConfig(output_dir=a, **cfg))
    builder_a.generate_dataset(6, parallel=False)

    builder_b = DatasetBuilder(GeneratorConfig(output_dir=b, **cfg))
    builder_b.generate_dataset(3, parallel=False, start_sample=0)
    builder_b.generate_dataset(3, parallel=False, start_sample=3)

    for i in range(6):
        fa = a / f"sample_{i:05d}.py"
        fb = b / f"sample_{i:05d}.py"
        assert fa.exists(), f"missing {fa}"
        assert fb.exists(), f"missing {fb}"
        assert fa.read_text(encoding="utf-8") == fb.read_text(encoding="utf-8"), \
            f"sample {i} differs between one-go and chunked runs"


def test_start_sample_full_mode_resume(tmp_path):
    """Возобновление в полном режиме: start_sample=2 пишет только сэмплы 2..3."""
    config = GeneratorConfig(output_dir=tmp_path,
                             complexity_levels=["easy"],
                             save_formats=["program_py", "step", "stl"])
    builder = DatasetBuilder(config)
    builder.generate_dataset(2, parallel=False, start_sample=2)
    for i in (2, 3):
        d = tmp_path / "easy" / f"sample_{i:05d}"
        assert (d / "program.py").exists()
        assert (d / "metadata.json").exists()
        assert (d / "ast.json").exists()
        assert (d / "model.step").exists()
        assert (d / "model.stl").exists()
    assert not (tmp_path / "easy" / "sample_00000").exists()
    assert not (tmp_path / "easy" / "sample_00001").exists()


def test_start_sample_flat_resume_continues(tmp_path):
    """Плоский режим: два прогона по 2 (0..1 и 2..3) дополняют одну папку."""
    config = GeneratorConfig(output_dir=tmp_path,
                             complexity_levels=["easy"],
                             split_by_complexity=False)
    builder = DatasetBuilder(config)
    builder.generate_dataset(2, parallel=False, start_sample=0)
    builder.generate_dataset(2, parallel=False, start_sample=2)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["sample_00000.py", "sample_00001.py",
                     "sample_00002.py", "sample_00003.py"]


def test_windows_do_not_change_dataset(tmp_path):
    """Оконная обработка (window_size=4 < батча) не меняет датасет:
    generate_dataset с окнами даёт ровно те же файлы, что прямое построение
    каждого сэмпла по его seed из SeedSequence."""
    import numpy as np
    cfg = dict(complexity_levels=["easy"], split_by_complexity=False, global_seed=7)
    a = tmp_path / "windowed"
    b = tmp_path / "direct"
    # 9 сэмплов при window_size=max(4,4*workers)=4 → три окна (4+4+1)
    DatasetBuilder(GeneratorConfig(output_dir=a, **cfg)).generate_dataset(9, parallel=False)

    states = np.random.SeedSequence(7).generate_state(9)
    builder_b = DatasetBuilder(GeneratorConfig(output_dir=b, **cfg))
    for i in range(9):
        builder_b.generate_sample(i, int(states[i]))

    a_files = {p.name: p.read_text(encoding="utf-8") for p in a.glob("*.py")}
    b_files = {p.name: p.read_text(encoding="utf-8") for p in b.glob("*.py")}
    assert set(a_files) == set(b_files), "windowed/direct file sets differ"
    for name in sorted(a_files):
        assert a_files[name] == b_files[name], f"{name} differs between windowed/direct"
