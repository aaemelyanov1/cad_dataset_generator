"""Тесты исполнителя: детерминированность, кэширование, распаковка в Solid."""
import numpy as np
import pytest

from cad_dataset_generator.builder.ast_builder import ASTBuilder
from cad_dataset_generator.config import GeneratorConfig
from cad_dataset_generator.executor.executor import Executor
from cad_dataset_generator.syntax_tree.nodes import PrimitiveNode


@pytest.fixture
def ast():
    config = GeneratorConfig()
    return ASTBuilder(config, seed=7, complexity="easy").build()


def test_execute_returns_solid(ast):
    shape = Executor().execute(ast)
    assert shape.ShapeType() == "Solid"
    assert float(shape.Volume()) > 0


def test_execute_caches_by_node_id(ast, strip_node_ids):
    ex = Executor()
    s1 = ex.execute(ast)
    assert len(ex.cache) == ast.count_operations()
    s2 = ex.execute(ast)
    assert s1 is s2


def test_execute_deterministic(strip_node_ids):
    config = GeneratorConfig()
    a1 = ASTBuilder(config, seed=99, complexity="medium").build()
    a2 = ASTBuilder(config, seed=99, complexity="medium").build()
    assert strip_node_ids(a1.to_dict()) == strip_node_ids(a2.to_dict())
    ex = Executor()
    v1 = float(ex.execute(a1).Volume())
    v2 = float(ex.execute(a2).Volume())
    assert abs(v1 - v2) < 1e-9


def test_bbox_cached():
    ex = Executor()
    node = PrimitiveNode(operation="box", parameters={"length": 2.0, "width": 3.0, "height": 4.0})
    bb1 = ex.bbox(node)
    bb2 = ex.bbox(node)
    assert bb1 == bb2
    assert bb1[1] - bb1[0] == pytest.approx(2.0, abs=1e-3)
    assert bb1[3] - bb1[2] == pytest.approx(3.0, abs=1e-3)
    assert bb1[5] - bb1[4] == pytest.approx(4.0, abs=1e-3)


def test_clear_empties_cache(ast):
    ex = Executor()
    ex.execute(ast)
    ex.clear()
    assert not ex.cache
    assert not ex._bbox_cache
