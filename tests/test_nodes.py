"""Тесты узлов синтаксического дерева: сериализация, счёт операций."""
import pytest
from cad_dataset_generator.syntax_tree.nodes import (
    PrimitiveNode, SketchNode, PathNode, ExtrudeNode, RevolveNode,
    LoftNode, SweepNode, FilletNode, ChamferNode, ShellNode, HoleNode,
    TransformNode, PatternNode, BooleanNode, ASTNode,
)


@pytest.mark.parametrize("cls,operation,params", [
    (PrimitiveNode, "box", {"length": 1.0, "width": 2.0, "height": 3.0}),
    (SketchNode, "rect", {"width": 2.0, "height": 2.0}),
    (PathNode, "path", {"points": [(0, 0, 0), (2, 0, 0)], "method": "spline"}),
])
def test_leaf_roundtrip(cls, operation, params):
    node = cls(operation=operation, parameters=params)
    restored = ASTNode._node_from_dict(node.to_dict())
    assert isinstance(restored, cls)
    assert restored.operation == operation
    assert restored.parameters == params


def test_extrude_roundtrip():
    sk = SketchNode(operation="rect", parameters={"width": 2.0, "height": 2.0})
    node = ExtrudeNode(operation="extrude", parameters={"distance": 5.0}, children=[sk])
    restored = ASTNode.from_json(node.to_json())
    assert isinstance(restored, ExtrudeNode)
    assert isinstance(restored.children[0], SketchNode)
    assert restored.children[0].operation == "rect"


def test_count_operations_ignores_sketches_and_paths():
    sk = SketchNode(operation="rect", parameters={})
    path = PathNode(operation="path", parameters={})
    sweep = SweepNode(operation="sweep", parameters={}, children=[sk, path])
    assert sweep.count_operations() == 1
    assert sweep.count_nodes() == 3
    assert sweep.all_operations() == ["sweep", "rect", "path"]


def test_boolean_children_required():
    with pytest.raises(ValueError):
        BooleanNode(operation="union", parameters={}, children=[PrimitiveNode(operation="box", parameters={})])


def test_unknown_operation_rejected():
    with pytest.raises(ValueError):
        ASTNode._node_from_dict({"operation": "nope", "parameters": {}, "children": []})
