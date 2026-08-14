from __future__ import annotations
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ASTNode(ABC):
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    operation: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    children: List[ASTNode] = field(default_factory=list)
    shape_state: str = "EMPTY"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "operation": self.operation,
            "parameters": self.parameters,
            "children": [child.to_dict() for child in self.children],
            "shape_state": self.shape_state,
            "metadata": self.metadata
        }

    @classmethod
    @abstractmethod
    def from_dict(cls, d: Dict[str, Any]) -> ASTNode:
        pass

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> ASTNode:
        d = json.loads(json_str)
        return cls._node_from_dict(d)

    @staticmethod
    def _node_from_dict(d: Dict[str, Any]) -> ASTNode:
        op = d["operation"]
        class_map = {
            "box": PrimitiveNode, "cylinder": PrimitiveNode, "sphere": PrimitiveNode,
            "cone": PrimitiveNode, "wedge": PrimitiveNode, "torus": PrimitiveNode,
            "rect": SketchNode, "circle": SketchNode, "ellipse": SketchNode,
            "polygon": SketchNode, "slot": SketchNode, "polyline": SketchNode,
            "spline": SketchNode,
            "roundrect": SketchNode, "frame": SketchNode, "sector": SketchNode,
            "arc_profile": SketchNode, "ellipse_arc": SketchNode,
            "bent": SketchNode, "mirrored": SketchNode,
            "extrude": ExtrudeNode, "revolve": RevolveNode,
            "twist_extrude": TwistExtrudeNode,
            "loft": LoftNode, "sweep": SweepNode,
            "path": PathNode,
            "fillet": FilletNode, "chamfer": ChamferNode,
            "shell": ShellNode, "hole": HoleNode,
            "split": SplitNode,
            "translate": TransformNode, "rotate": TransformNode, "mirror": TransformNode,
            "rarray": PatternNode, "polarArray": PatternNode,
            "union": BooleanNode, "cut": BooleanNode, "intersect": BooleanNode
        }
        node_cls = class_map.get(op)
        if node_cls is None:
            raise ValueError(f"Unknown operation: {op}")
        return node_cls.from_dict(d)

    def count_nodes(self) -> int:
        """Количество узлов в поддереве, включая текущий."""
        return 1 + sum(child.count_nodes() for child in self.children)

    def count_operations(self) -> int:
        """Количество «операций» (не эскизов и не путей) в поддереве.

        Примитивы, extrude/revolve/loft/sweep, модификаторы, булевы операции и
        трансформации считаются как 1 операция каждая; вспомогательные узлы
        (SketchNode, PathNode) в счёт не идут.
        """
        if isinstance(self, SketchNode) or isinstance(self, PathNode):
            return 0
        return 1 + sum(child.count_operations() for child in self.children)

    def all_operations(self) -> List[str]:
        """Список операций всех узлов поддерева (включая эскизы/пути)."""
        ops = [self.operation]
        for child in self.children:
            ops.extend(child.all_operations())
        return ops

@dataclass
class PrimitiveNode(ASTNode):
    def __post_init__(self):
        self.shape_state = "SOLID"

    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PrimitiveNode:
        return cls(node_id=d["node_id"], operation=d["operation"],
                   parameters=d["parameters"], shape_state=d["shape_state"],
                   metadata=d.get("metadata", {}))

@dataclass
class SketchNode(ASTNode):
    def __post_init__(self):
        self.shape_state = "SKETCH"

    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SketchNode:
        return cls(node_id=d["node_id"], operation=d["operation"],
                   parameters=d["parameters"], shape_state=d["shape_state"],
                   metadata=d.get("metadata", {}))

@dataclass
class ExtrudeNode(ASTNode):
    def __post_init__(self):
        self.shape_state = "SOLID"
        if len(self.children) != 1 or not isinstance(self.children[0], SketchNode):
            raise ValueError("Extrude requires one SketchNode child")

    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExtrudeNode:
        child = ASTNode._node_from_dict(d["children"][0])
        return cls(node_id=d["node_id"], operation=d["operation"],
                   parameters=d["parameters"], children=[child],
                   shape_state=d["shape_state"], metadata=d.get("metadata", {}))

@dataclass
class RevolveNode(ASTNode):
    def __post_init__(self):
        self.shape_state = "SOLID"
        if len(self.children) != 1 or not isinstance(self.children[0], SketchNode):
            raise ValueError("Revolve requires one SketchNode child")

    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RevolveNode:
        child = ASTNode._node_from_dict(d["children"][0])
        return cls(node_id=d["node_id"], operation=d["operation"],
                   parameters=d["parameters"], children=[child],
                   shape_state=d["shape_state"], metadata=d.get("metadata", {}))

@dataclass
class TwistExtrudeNode(ASTNode):
    def __post_init__(self):
        self.shape_state = "SOLID"
        if len(self.children) != 1 or not isinstance(self.children[0], SketchNode):
            raise ValueError("TwistExtrude requires one SketchNode child")

    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TwistExtrudeNode:
        child = ASTNode._node_from_dict(d["children"][0])
        return cls(node_id=d["node_id"], operation=d["operation"],
                   parameters=d["parameters"], children=[child],
                   shape_state=d["shape_state"], metadata=d.get("metadata", {}))


@dataclass
class LoftNode(ASTNode):
    def __post_init__(self):
        self.shape_state = "SOLID"
        if len(self.children) < 2 or not all(isinstance(c, SketchNode) for c in self.children):
            raise ValueError("Loft requires at least two SketchNode children")

    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LoftNode:
        children = [ASTNode._node_from_dict(c) for c in d["children"]]
        return cls(node_id=d["node_id"], operation=d["operation"],
                   parameters=d["parameters"], children=children,
                   shape_state=d["shape_state"], metadata=d.get("metadata", {}))

@dataclass
class SweepNode(ASTNode):
    def __post_init__(self):
        self.shape_state = "SOLID"
        if len(self.children) != 2:
            raise ValueError("Sweep requires two children: profile SketchNode and path SketchNode")
        if not isinstance(self.children[0], SketchNode) or not isinstance(self.children[1], (SketchNode, PathNode)):
            raise ValueError("Invalid children types for Sweep")

    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SweepNode:
        children = [ASTNode._node_from_dict(c) for c in d["children"]]
        return cls(node_id=d["node_id"], operation=d["operation"],
                   parameters=d["parameters"], children=children,
                   shape_state=d["shape_state"], metadata=d.get("metadata", {}))

@dataclass
class PathNode(ASTNode):
    def __post_init__(self):
        self.shape_state = "SKETCH"
    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PathNode:
        return cls(node_id=d["node_id"], operation=d["operation"],
                   parameters=d["parameters"], shape_state=d["shape_state"],
                   metadata=d.get("metadata", {}))

@dataclass
class FilletNode(ASTNode):
    def __post_init__(self):
        self.shape_state = "SOLID"
        if len(self.children) != 1:
            raise ValueError("Fillet requires one solid child")
    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FilletNode:
        child = ASTNode._node_from_dict(d["children"][0])
        return cls(node_id=d["node_id"], operation=d["operation"],
                   parameters=d["parameters"], children=[child],
                   shape_state=d["shape_state"], metadata=d.get("metadata", {}))

@dataclass
class ChamferNode(ASTNode):
    def __post_init__(self):
        self.shape_state = "SOLID"
        if len(self.children) != 1:
            raise ValueError("Chamfer requires one solid child")
    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ChamferNode:
        child = ASTNode._node_from_dict(d["children"][0])
        return cls(node_id=d["node_id"], operation=d["operation"],
                   parameters=d["parameters"], children=[child],
                   shape_state=d["shape_state"], metadata=d.get("metadata", {}))

@dataclass
class ShellNode(ASTNode):
    def __post_init__(self):
        self.shape_state = "SOLID"
        if len(self.children) != 1:
            raise ValueError("Shell requires one solid child")
    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ShellNode:
        child = ASTNode._node_from_dict(d["children"][0])
        return cls(node_id=d["node_id"], operation=d["operation"],
                   parameters=d["parameters"], children=[child],
                   shape_state=d["shape_state"], metadata=d.get("metadata", {}))

@dataclass
class HoleNode(ASTNode):
    def __post_init__(self):
        self.shape_state = "SOLID"
        if len(self.children) != 1:
            raise ValueError("Hole requires one solid child")
    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> HoleNode:
        child = ASTNode._node_from_dict(d["children"][0])
        return cls(node_id=d["node_id"], operation=d["operation"],
                   parameters=d["parameters"], children=[child],
                   shape_state=d["shape_state"], metadata=d.get("metadata", {}))


@dataclass
class SplitNode(ASTNode):
    def __post_init__(self):
        self.shape_state = "SOLID"
        if len(self.children) != 1:
            raise ValueError("Split requires one solid child")
    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SplitNode:
        child = ASTNode._node_from_dict(d["children"][0])
        return cls(node_id=d["node_id"], operation=d["operation"],
                   parameters=d["parameters"], children=[child],
                   shape_state=d["shape_state"], metadata=d.get("metadata", {}))

@dataclass
class TransformNode(ASTNode):
    def __post_init__(self):
        self.shape_state = "SOLID"
        if len(self.children) != 1:
            raise ValueError("Transform requires one solid child")
    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TransformNode:
        child = ASTNode._node_from_dict(d["children"][0])
        return cls(node_id=d["node_id"], operation=d["operation"],
                   parameters=d["parameters"], children=[child],
                   shape_state=d["shape_state"], metadata=d.get("metadata", {}))

@dataclass
class PatternNode(ASTNode):
    def __post_init__(self):
        self.shape_state = "SOLID"
        if len(self.children) != 1:
            raise ValueError("Pattern requires one solid child")
    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PatternNode:
        child = ASTNode._node_from_dict(d["children"][0])
        return cls(node_id=d["node_id"], operation=d["operation"],
                   parameters=d["parameters"], children=[child],
                   shape_state=d["shape_state"], metadata=d.get("metadata", {}))

@dataclass
class BooleanNode(ASTNode):
    def __post_init__(self):
        self.shape_state = "SOLID"
        if len(self.children) != 2:
            raise ValueError("Boolean requires two solid children")
    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BooleanNode:
        children = [ASTNode._node_from_dict(c) for c in d["children"]]
        return cls(node_id=d["node_id"], operation=d["operation"],
                   parameters=d["parameters"], children=children,
                   shape_state=d["shape_state"], metadata=d.get("metadata", {}))