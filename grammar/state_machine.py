from enum import Enum
from typing import List, Dict

class ShapeState(str, Enum):
    EMPTY = "EMPTY"
    SKETCH = "SKETCH"
    SOLID = "SOLID"

_TRANSITIONS: Dict[ShapeState, List[str]] = {
    ShapeState.EMPTY: [
        "box", "cylinder", "sphere", "cone", "wedge", "torus",
        "rect", "circle", "ellipse", "polygon", "slot", "polyline", "spline",
        "roundrect", "frame", "sector", "arc_profile", "ellipse_arc",
        "bent", "mirrored"
    ],
    ShapeState.SKETCH: ["extrude", "revolve", "twist_extrude", "loft", "sweep"],
    ShapeState.SOLID: [
        "fillet", "chamfer", "shell", "hole", "split",
        "translate", "rotate", "mirror",
        "rarray", "polarArray", "scatter",
        "union", "cut", "intersect"
    ]
}

def get_allowed_operations(state: ShapeState) -> List[str]:
    return _TRANSITIONS.get(state, [])