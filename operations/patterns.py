"""Массивы: прямоугольный rarray и полярный polarArray через объединение копий."""
import cadquery as cq

from ..exceptions import GenerationError
from .booleans import union


def rectangular_array(
    solid: cq.Shape, nx: int, ny: int, spacing_x: float, spacing_y: float
) -> cq.Shape:
    if nx < 1 or ny < 1:
        raise GenerationError("rarray requires nx, ny >= 1")
    result = solid
    for i in range(nx):
        for j in range(ny):
            if i == 0 and j == 0:
                continue
            offset = cq.Vector(i * spacing_x, j * spacing_y, 0.0)
            result = union(result, solid.translate(offset))
    return result


def polar_array(solid: cq.Shape, count: int, angle: float = 360.0) -> cq.Shape:
    if count < 1:
        raise GenerationError("polarArray requires count >= 1")
    center = solid.Center()
    axis_start = center
    axis_end = center + cq.Vector(0, 0, 1) * 10.0  # ось Z через центр
    result = solid
    for i in range(1, count):
        rotated = solid.rotate(axis_start, axis_end, angle * i / count)
        result = union(result, rotated)
    return result
