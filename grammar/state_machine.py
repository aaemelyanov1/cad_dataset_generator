from enum import Enum
from typing import List, Dict

class ShapeState(str, Enum):
    EMPTY = "EMPTY"
    SKETCH = "SKETCH"
    SOLID = "SOLID"

_TRANSITIONS: Dict[ShapeState, List[str]] = {
    ShapeState.EMPTY: [
        "box", "cylinder", "sphere", "cone", "wedge", "torus",
        "rect", "circle", "ellipse", "polygon", "slot", "polyline", "spline"
    ],
    ShapeState.SKETCH: ["extrude", "revolve", "loft", "sweep"],
    ShapeState.SOLID: [
        "fillet", "chamfer", "shell", "hole",
        "translate", "rotate", "mirror",
        "rarray", "polarArray",
        "union", "cut", "intersect"
    ]
}

def get_allowed_operations(state: ShapeState) -> List[str]:
    return _TRANSITIONS.get(state, [])