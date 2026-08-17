"""Исполнение узлов AST в геометрию CadQuery.

Исполнитель детерминирован: каждый узел однозначно отображается в cq.Shape,
а параметры (радиусы скруглений, толщины и т.п.) уже зафиксированы в узле
на этапе построения AST.
"""
import cadquery as cq
from typing import Dict

from ..syntax_tree.nodes import (
    ASTNode, PrimitiveNode, SketchNode, ExtrudeNode, RevolveNode,
    TwistExtrudeNode, LoftNode, SweepNode, FilletNode, ChamferNode,
    ShellNode, HoleNode, SplitNode, TransformNode, PatternNode, BooleanNode,
)
from ..operations import (
    make_box, make_cylinder, make_sphere, make_cone, make_wedge, make_torus,
    extrude, revolve, twist_extrude, loft, sweep,
    fillet, chamfer, shell, hole, split,
    translate, rotate, mirror,
    rectangular_array, polar_array, scatter,
    union, cut, intersect,
)
from ..utils.geometry_utils import unwrap_solid, fast_bbox
from ..exceptions import GenerationError


class Executor:
    def __init__(self):
        self.cache: Dict[str, cq.Shape] = {}
        self._bbox_cache: Dict[str, tuple] = {}

    def clear(self):
        self.cache.clear()
        self._bbox_cache.clear()

    def execute(self, node: ASTNode) -> cq.Shape:
        if node.node_id in self.cache:
            return self.cache[node.node_id]
        shape = self._dispatch(node)
        shape = unwrap_solid(shape)
        self.cache[node.node_id] = shape
        return shape

    def bbox(self, node: ASTNode) -> tuple:
        """Кэшированный быстрый bbox результата узла."""
        if node.node_id in self._bbox_cache:
            return self._bbox_cache[node.node_id]
        bb = fast_bbox(self.execute(node))
        self._bbox_cache[node.node_id] = bb
        return bb

    def _dispatch(self, node: ASTNode) -> cq.Shape:
        if isinstance(node, PrimitiveNode):
            return self._primitive(node)
        if isinstance(node, SketchNode):
            raise GenerationError("SketchNode cannot be executed directly.")
        if isinstance(node, ExtrudeNode):
            sk = node.children[0]
            return extrude(sk.operation, sk.parameters, node.parameters["distance"])
        if isinstance(node, RevolveNode):
            sk = node.children[0]
            return revolve(sk.operation, sk.parameters, node.parameters.get("angle", 360.0))
        if isinstance(node, TwistExtrudeNode):
            sk = node.children[0]
            return twist_extrude(sk.operation, sk.parameters,
                                 node.parameters["distance"],
                                 node.parameters.get("angle", 0.0))
        if isinstance(node, LoftNode):
            profiles = [(c.operation, c.parameters) for c in node.children]
            return loft(profiles, node.parameters["offsets"])
        if isinstance(node, SweepNode):
            profile = node.children[0]
            path = node.children[1]
            return sweep(profile.operation, profile.parameters, path.parameters)
        if isinstance(node, FilletNode):
            child = self.execute(node.children[0])
            return fillet(child, node.parameters["radius"], node.parameters.get("selection"))
        if isinstance(node, ChamferNode):
            child = self.execute(node.children[0])
            return chamfer(child, node.parameters["distance"], node.parameters.get("selection"))
        if isinstance(node, ShellNode):
            child = self.execute(node.children[0])
            return shell(child, node.parameters["thickness"], node.parameters.get("selection"))
        if isinstance(node, HoleNode):
            child = self.execute(node.children[0])
            p = node.parameters
            return hole(child, p["position"], p["radius"], p["depth"],
                        kind=p.get("kind", "through"),
                        cbo_radius=p.get("cbo_radius"), cbo_depth=p.get("cbo_depth"),
                        csk_radius=p.get("csk_radius"), csk_depth=p.get("csk_depth"))
        if isinstance(node, SplitNode):
            child = self.execute(node.children[0])
            return split(child, node.parameters["axis"], node.parameters["gap"],
                         plane_offset=node.parameters.get("plane_offset"),
                         parts_count=node.parameters.get("parts_count"))
        if isinstance(node, TransformNode):
            child = self.execute(node.children[0])
            op = node.operation
            if op == "translate":
                return translate(child, node.parameters["vector"])
            if op == "rotate":
                return rotate(child, node.parameters["axis"], node.parameters["angle"])
            if op == "mirror":
                return mirror(child, node.parameters["plane"])
            raise GenerationError(f"Unknown transform {op}")
        if isinstance(node, PatternNode):
            child = self.execute(node.children[0])
            op = node.operation
            if op == "rarray":
                return rectangular_array(child, node.parameters["nx"], node.parameters["ny"],
                                        node.parameters["spacing_x"], node.parameters["spacing_y"])
            if op == "polarArray":
                return polar_array(child, node.parameters["count"],
                                  node.parameters["radius"],
                                  node.parameters.get("angle", 360.0),
                                  node.parameters.get("start_angle", 0.0),
                                  node.parameters.get("fill", True))
            if op == "scatter":
                return scatter(child, node.parameters["points"])
            raise GenerationError(f"Unknown pattern {op}")
        if isinstance(node, BooleanNode):
            left = self.execute(node.children[0])
            right = self.execute(node.children[1])
            offset = node.parameters.get("offset", (0.0, 0.0, 0.0))
            if any(float(o) != 0.0 for o in offset):
                right = translate(right, offset)
            op = node.operation
            if op == "union":
                return union(left, right)
            if op == "cut":
                return cut(left, right)
            if op == "intersect":
                return intersect(left, right)
            raise GenerationError(f"Unknown boolean {op}")
        raise GenerationError(f"Unknown node type {type(node)}")

    def _primitive(self, node: PrimitiveNode) -> cq.Shape:
        p = node.parameters
        op = node.operation
        if op == "box":
            return make_box(p["length"], p["width"], p["height"])
        if op == "cylinder":
            return make_cylinder(p["height"], p["radius"])
        if op == "sphere":
            return make_sphere(p["radius"])
        if op == "cone":
            return make_cone(p["height"], p["radius1"], p["radius2"])
        if op == "wedge":
            return make_wedge(p["dx"], p["dy"], p["dz"], p["xmin"], p["zmin"],
                              p["xmax"], p["zmax"])
        if op == "torus":
            return make_torus(p["radius1"], p["radius2"])
        raise GenerationError(f"Unknown primitive {op}")
