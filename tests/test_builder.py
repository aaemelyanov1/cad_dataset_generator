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


def test_new_ops_diversity():
    """За достаточное число попыток должны встретиться twist_extrude, split и виды отверстий."""
    config = GeneratorConfig()
    seen = set()
    kinds = set()
    for seed in range(80):
        ast = ASTBuilder(config, seed=seed, complexity="medium").build()
        seen.update(ast.all_operations())

        def walk(n):
            if n.operation == "hole":
                kinds.add(n.parameters.get("kind", "through"))
            for c in getattr(n, "children", []) or []:
                walk(c)
        walk(ast)
    assert "twist_extrude" in seen, f"twist_extrude не встретился; seen={sorted(seen)}"
    assert "split" in seen, f"split не встретился; seen={sorted(seen)}"
    assert kinds == {"through", "blind", "cbore", "csk"}, f"hole kinds: {kinds}"


def test_profile_rebalance_keeps_groups():
    """Ребаланс весов не убирает редкие группы: за 60 medium должны быть
    shell, массивы и loft/sweep, а transforms не должны доминировать."""
    config = GeneratorConfig()
    hist = {}
    for seed in range(60):
        ast = ASTBuilder(config, seed=seed, complexity="medium").build()
        for op in ast.all_operations():
            hist[op] = hist.get(op, 0) + 1
    total = sum(hist.values())
    assert total > 0
    transforms = sum(hist.get(op, 0) for op in ("translate", "rotate", "mirror"))
    assert transforms / total < 0.5, f"transforms доминируют: {transforms / total:.2f}"
    assert hist.get("shell", 0) > 0, "shell исчез после ребаланса"
    assert hist.get("rarray", 0) + hist.get("polarArray", 0) + hist.get("scatter", 0) > 0
    assert hist.get("loft", 0) + hist.get("sweep", 0) > 0


def test_medium_build_time_budget():
    """Базовый временной бюджет: medium-сборка укладывается в 5с (до фикса было ~1-3.5с)"""
    import time
    config = GeneratorConfig()
    t0 = time.time()
    for seed in range(5):
        ASTBuilder(config, seed=seed, complexity="medium").build()
    elapsed = time.time() - t0
    per = elapsed / 5
    assert per < 5.0, f"medium build too slow: {per:.2f}s/sample"
