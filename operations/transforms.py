"""Пространственные операции: перенос, вращение, зеркалирование."""
import cadquery as cq

from ..exceptions import GenerationError


def translate(solid: cq.Shape, vector) -> cq.Shape:
    return solid.translate(cq.Vector(*vector))


def rotate(solid: cq.Shape, axis: str, angle: float) -> cq.Shape:
    """Вращение твёрдого тела вокруг оси, проходящей через его центр."""
    axis_map = {
        "X": cq.Vector(1, 0, 0),
        "Y": cq.Vector(0, 1, 0),
        "Z": cq.Vector(0, 0, 1),
    }
    if axis not in axis_map:
        raise GenerationError(f"Unknown rotation axis: {axis}")
    center = solid.Center()
    axis_end = center + axis_map[axis] * 10.0
    try:
        return solid.rotate(center, axis_end, angle)
    except Exception as e:
        raise GenerationError(f"rotate failed: {e}") from e


def mirror(solid: cq.Shape, plane: str) -> cq.Shape:
    """Зеркалирование относительно плоскости, проходящей через центр тела."""
    center = solid.Center()
    try:
        return solid.mirror(plane, center)
    except Exception as e:
        raise GenerationError(f"mirror failed: {e}") from e
