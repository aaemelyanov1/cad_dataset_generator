import sys
from pathlib import Path
# Добавляем корень проекта в путь, если запускаем напрямую
sys.path.insert(0, str(Path(__file__).parent.parent))
from cad_dataset_generator.config import GeneratorConfig
from cad_dataset_generator.dataset.dataset_builder import DatasetBuilder
from cad_dataset_generator.utils.logging import setup_logging
import argparse

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Генератор датасета 3D моделей для RLVR")
    parser.add_argument("--num_samples", type=int, default=100, help="Количество моделей")
    parser.add_argument("--output", type=str, default="output", help="Папка для сохранения")
    parser.add_argument("--seed", type=int, default=42, help="Глобальный seed")
    parser.add_argument("--parallel", action="store_true", help="Параллельная генерация")
    parser.add_argument("--programs-only", action="store_true",
                        help="Сохранять только программы (без render/stl/step/pointcloud/mesh/ast_json)")
    args = parser.parse_args()
    config = GeneratorConfig(
        global_seed=args.seed,
        output_dir=Path(args.output)
    )
    if args.programs_only:
        config.save_formats = ["program_py", "program_txt"]
    builder = DatasetBuilder(config)
    builder.generate_dataset(args.num_samples, parallel=args.parallel)

if __name__ == "__main__":
    main()
    import sys
    sys.exit(0)