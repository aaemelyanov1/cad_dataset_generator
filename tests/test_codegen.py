"""Тесты генерации кода: program.py воспроизводит геометрию исполнителя.

Ключевые требования:
  * в теле программы нет пользовательских функций (def) и хелперов;
  * программа состоит только из вызовов нативного API CadQuery;
  * каждая отдельная операция воспроизводит геометрию Executor'а;
  * сгенерированные программы используют эскизы, а не только примитивы.
"""
import numpy as np
import pytest

from cad_dataset_generator.builder.ast_builder import ASTBuilder
from cad_dataset_generator.code_generator import generate_code
from cad_dataset_generator.config import GeneratorConfig
from cad_dataset_generator.executor.executor import Executor
from cad_dataset_generator.syntax_tree.nodes import (
    ASTNode, PrimitiveNode, SketchNode, PathNode, ExtrudeNode, RevolveNode,
    LoftNode, SweepNode, FilletNode, ChamferNode,
    ShellNode, HoleNode, TransformNode, PatternNode, BooleanNode,
)


def _P(op, **params):
    return PrimitiveNode(operation=op, parameters=params)


def _S(op, **params):
    return SketchNode(operation=op, parameters=params)


def _run(ast):
    code = generate_code(ast)
    ns = {}
    exec(code, ns)
    return ns["result"], code


def _bbox(shape):
    bb = shape.BoundingBox()
    return (bb.xmax, bb.ymax, bb.zmax, bb.xmin, bb.ymin, bb.zmin)


def _assert_reproduces(ast, shape):
    result, code = _run(ast)
    assert float(result.Volume()) == pytest.approx(float(shape.Volume()), rel=1e-6, abs=1e-6)
    assert _bbox(result) == pytest.approx(_bbox(shape), rel=1e-6, abs=1e-6)
    return code


# --------------------------------------------------------------------------- #
# Поштучная проверка операций: код каждой операции воспроизводит Executor      #
# --------------------------------------------------------------------------- #
def _individual_ops():
    return [
        ("box", _P("box", length=2.0, width=3.0, height=4.0)),
        ("cylinder", _P("cylinder", height=4.0, radius=1.5)),
        ("sphere", _P("sphere", radius=2.0)),
        ("cone", _P("cone", radius1=2.0, radius2=0.5, height=4.0)),
        ("wedge", _P("wedge", dx=3.0, dy=4.0, dz=5.0, xmin=0.0, zmin=0.0,
                     xmax=2.0, zmax=3.0)),
        ("torus", _P("torus", radius1=3.0, radius2=0.5)),
        ("extrude_rect", ExtrudeNode(operation="extrude", parameters={"distance": 3.0},
                                     children=[_S("rect", workplane="XY", center=(0.0, 0.0),
                                                  width=2.0, height=3.0)])),
        ("extrude_circle", ExtrudeNode(operation="extrude", parameters={"distance": 3.0},
                                       children=[_S("circle", workplane="XY", center=(0.0, 0.0),
                                                    radius=1.5)])),
        ("extrude_polyline", ExtrudeNode(
            operation="extrude", parameters={"distance": 3.0},
            children=[_S("polyline", workplane="XY", center=(0.0, 0.0),
                         points=[(2.0, 0.0), (0.0, 2.0), (-1.0, 0.0), (0.0, -1.5)])])),
        ("extrude_spline", ExtrudeNode(
            operation="extrude", parameters={"distance": 3.0},
            children=[_S("spline", workplane="XY", center=(0.0, 0.0),
                         points=[(2.0, 0.0), (0.0, 2.0), (-1.0, 0.0), (0.0, -1.5)])])),
        ("extrude_slot", ExtrudeNode(operation="extrude", parameters={"distance": 3.0},
                                     children=[_S("slot", workplane="XY", center=(0.0, 0.0),
                                                  length=4.0, width=1.0)])),
        ("revolve_rect", RevolveNode(operation="revolve", parameters={"angle": 360.0},
                                     children=[_S("rect", workplane="XY", center=(4.0, 0.0),
                                                  width=2.0, height=3.0)])),
        ("loft", LoftNode(operation="loft", parameters={"offsets": [0.0, 5.0]},
                          children=[_S("circle", workplane="XY", center=(0.0, 0.0), radius=2.0),
                                    _S("circle", workplane="XY", center=(0.0, 0.0), radius=1.0)])),
        ("sweep", SweepNode(operation="sweep", parameters={},
                            children=[_S("circle", workplane="XY", center=(0.0, 0.0), radius=0.5),
                                      PathNode(operation="path", parameters={
                                          "workplane": "XY",
                                          "points": [(0.0, 0.0, 0.0), (2.0, 0.0, 0.5), (2.0, 2.0, 1.0)],
                                          "method": "polyline",
                                      })])),
        ("fillet", FilletNode(operation="fillet", parameters={"radius": 0.4},
                              children=[_P("box", length=2.0, width=3.0, height=4.0)])),
        ("chamfer", ChamferNode(operation="chamfer", parameters={"distance": 0.4},
                                children=[_P("box", length=2.0, width=3.0, height=4.0)])),
        ("shell", ShellNode(operation="shell", parameters={"thickness": 0.2, "face_rank": 0},
                            children=[_P("box", length=4.0, width=4.0, height=4.0)])),
        ("hole", HoleNode(operation="hole",
                          parameters={"position": (0.0, 0.0, -1.0), "radius": 0.5, "depth": 6.0},
                          children=[_P("box", length=4.0, width=4.0, height=4.0)])),
        ("translate", TransformNode(operation="translate", parameters={"vector": (3.0, 4.0, 5.0)},
                                    children=[_P("box", length=2.0, width=2.0, height=2.0)])),
        ("rotate", TransformNode(operation="rotate", parameters={"axis": "Z", "angle": 45.0},
                                 children=[_P("box", length=2.0, width=2.0, height=2.0)])),
        ("mirror", TransformNode(operation="mirror", parameters={"plane": "XY"},
                                 children=[_P("box", length=2.0, width=2.0, height=2.0)])),
        ("rarray", PatternNode(operation="rarray",
                               parameters={"nx": 3, "ny": 2, "spacing_x": 1.5, "spacing_y": 1.5},
                               children=[_P("box", length=2.0, width=2.0, height=2.0)])),
        ("polarArray", PatternNode(operation="polarArray", parameters={"count": 4, "angle": 360.0},
                                   children=[_P("box", length=2.0, width=2.0, height=2.0)])),
        ("union", BooleanNode(operation="union", parameters={},
                              children=[_P("box", length=2.0, width=2.0, height=2.0),
                                        TransformNode(operation="translate",
                                                      parameters={"vector": (1.5, 0.0, 0.0)},
                                                      children=[_P("box", length=2.0, width=2.0, height=2.0)])])),
        ("cut", BooleanNode(operation="cut", parameters={},
                            children=[_P("box", length=2.0, width=2.0, height=2.0),
                                      TransformNode(operation="translate",
                                                    parameters={"vector": (1.0, 0.0, 0.0)},
                                                    children=[_P("box", length=2.0, width=2.0, height=2.0)])])),
        ("intersect", BooleanNode(operation="intersect", parameters={},
                                  children=[_P("box", length=2.0, width=2.0, height=2.0),
                                            TransformNode(operation="translate",
                                                          parameters={"vector": (1.0, 0.0, 0.0)},
                                                          children=[_P("box", length=2.0, width=2.0, height=2.0)])])),
    ]


@pytest.mark.parametrize("name,ast", _individual_ops(), ids=[n for n, _ in _individual_ops()])
def test_individual_operation_reproduces_executor(name, ast):
    shape = Executor().execute(ast)
    _assert_reproduces(ast, shape)


# --------------------------------------------------------------------------- #
# Нет функций и хелперов в сгенерированной программе                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("complexity", ["easy", "medium", "hard"])
def test_generated_code_has_no_functions(complexity):
    config = GeneratorConfig()
    builder = ASTBuilder(config, seed=7, complexity=complexity)
    ast = builder.build()
    code = generate_code(ast)
    assert "def " not in code
    assert code.startswith("import cadquery as cq")
    assert "_unwrap_solid" not in code
    assert "_edge_list" not in code
    assert "_sorted_faces" not in code


# --------------------------------------------------------------------------- #
# В программах используются эскизы, а не только примитивы                     #
# --------------------------------------------------------------------------- #
def test_programs_use_sketch_operations():
    """За достаточное число программ должны встретиться extrude/revolve/loft/sweep."""
    config = GeneratorConfig()
    sketch_ops_seen = set()
    primitives_only = 0
    for seed in range(60):
        ast = ASTBuilder(config, seed=seed, complexity="medium").build()
        sketch_ops_seen.update(ast.all_operations())
    assert {"extrude", "revolve", "loft", "sweep"} & sketch_ops_seen
    assert {"rect", "circle", "polyline", "spline", "polygon", "slot"} & sketch_ops_seen
