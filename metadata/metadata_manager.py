# ==============================
# metadata/metadata_manager.py
# ==============================
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
from ..syntax_tree.nodes import ASTNode


class Metadata(BaseModel):
    sample_id: str
    complexity: str
    seed: int
    operations_count: int
    ast_depth: int
    volume: float
    bounding_box_diagonal: float
    node_types: List[str] = Field(default_factory=list)
    config_info: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_ast(cls, ast: ASTNode, complexity: str, seed: int, config,
                 shape=None) -> 'Metadata':
        """Строит метаданные из AST; при наличии shape — с реальными габаритами."""
        if shape is not None:
            vol = float(shape.Volume())
            # точный bbox (не mesh-приближение) — не зависит от текущей триангуляции
            bb = shape.BoundingBox()
            diag = float(np.sqrt((bb.xmax - bb.xmin) ** 2
                                 + (bb.ymax - bb.ymin) ** 2
                                 + (bb.zmax - bb.zmin) ** 2))
        else:
            vol = float(ast.metadata.get("volume", 0.0))
            diag = float(ast.metadata.get("bounding_box_diagonal", 0.0))
        return cls(
            sample_id=ast.node_id,
            complexity=complexity,
            seed=seed,
            operations_count=ast.count_operations(),
            ast_depth=cls._depth(ast),
            volume=vol,
            bounding_box_diagonal=diag,
            node_types=ast.all_operations(),
            config_info=config.__dict__
        )

    @staticmethod
    def _depth(node: ASTNode) -> int:
        if not node.children:
            return 1
        return 1 + max(Metadata._depth(c) for c in node.children)
