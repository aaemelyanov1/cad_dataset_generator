"""Тесты построителя AST: диапазоны сложности, воспроизводимость, разнообразие операций."""
import pytest

from cad_dataset_generator.builder.ast_builder import ASTBuilder
from cad_dataset_generator.config import GeneratorConfig

OP_RANGES = {
    "easy": (3, 6),
    "medium": (7, 12),
    "hard": (13, 20),
    "expert": (20, 35),
}


@pytest.mark.parametrize("complexity", ["easy", "medium", "hard", "expert"])
def test_operations_within_range(complexity):
    config = GeneratorConfig()
    builder = ASTBuilder(config, seed=1, complexity=complexity)
    ast = builder.build()
    lo, hi = OP_RANGES[complexity]
    n = ast.count_operations()
    assert lo <= n <= hi, f"{complexity}: got {n} operations, expected [{lo}, {hi}]"


def test_reproducible_same_seed(strip_node_ids):
    config = GeneratorConfig()
    a = ASTBuilder(config, seed=123, complexity="hard").build()
    b = ASTBuilder(config, seed=123, complexity="hard").build()
    assert strip_node_ids(a.to_dict()) == strip_node_ids(b.to_dict())


def test_different_seed_different_tree():
    config = GeneratorConfig()
    a = ASTBuilder(config, seed=1, complexity="medium").build()
    b = ASTBuilder(config, seed=2, complexity="medium").build()
    assert a.to_json() != b.to_json()


def test_type_diversity():
    """За 40 попыток должны встретиться разные типы операций (не только примитивы)."""
    config = GeneratorConfig()
    seen = set()
    for seed in range(40):
        ast = ASTBuilder(config, seed=seed, complexity="medium").build()
        seen.update(ast.all_operations())
    assert {"box", "cylinder", "extrude"} & seen
    assert {"fillet", "chamfer", "shell", "hole", "union", "cut", "rarray"} & seen
