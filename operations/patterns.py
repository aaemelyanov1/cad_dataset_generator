"""Массивы: прямоугольный rarray, полярный polarArray и точечный scatter.

Все паттерны строятся нативными бесцикловыми вызовами CadQuery
(rarray/polarArray/pushPoints + eachpoint с combine="a" — объединение),
поэтому сгенерированная программа в точности повторяет геометрию исполнителя.
"""
import cadquery as cq
from typing import Any, Dict, List, Sequence

from ..exceptions import GenerationError


def _locate(solid: cq.Shape, center, op: str) -> cq.Workplane:
    """Рабочая плоскость с центром в `center`, готовая к раскладке копий."""
    try:
        return cq.Workplane("XY", origin=center)
    except Exception as e:
        raise GenerationError(f"{op} workplane failed: {e}") from e


def rectangular_array(
    solid: cq.Shape, nx: int, ny: int, spacing_x: float, spacing_y: float
) -> cq.Shape:
    if nx < 1 or ny < 1:
        raise GenerationError("rarray requires nx, ny >= 1")
    try:
        return (_locate(solid, solid.Center(), "rarray")
                .rarray(spacing_x, spacing_y, nx, ny, center=False)
                .eachpoint(solid, combine="a").val())
    except GenerationError:
        raise
    except Exception as e:
        raise GenerationError(f"rarray failed: {e}") from e


def polar_array(
    solid: cq.Shape,
    count: int,
    radius: float,
    angle: float = 360.0,
    start_angle: float = 0.0,
    fill: bool = True,
) -> cq.Shape:
    """Круговое размещение копий вокруг центра тела (bolt-circle).

    При `fill=True` первая позиция — центр, остальные — по окружности.
    """
    if count < 1:
        raise GenerationError("polarArray requires count >= 1")
    try:
        return (_locate(solid, solid.Center(), "polarArray")
                .polarArray(radius, start_angle, angle, count, fill=fill)
                .eachpoint(solid, combine="a").val())
    except GenerationError:
        raise
    except Exception as e:
        raise GenerationError(f"polarArray failed: {e}") from e


def scatter(solid: cq.Shape, points: Sequence[Sequence[float]]) -> cq.Shape:
    """Размещение копий по произвольным 2D-точкам (в плоскости XY, от центра)."""
    pts = [tuple(float(c) for c in p) for p in points]
    if not pts:
        raise GenerationError("scatter requires at least one point")
    try:
        return (_locate(solid, solid.Center(), "scatter")
                .pushPoints(pts)
                .eachpoint(solid, combine="a").val())
    except GenerationError:
        raise
    except Exception as e:
        raise GenerationError(f"scatter failed: {e}") from e
