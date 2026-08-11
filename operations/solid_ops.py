"""Твёрдотельные операции: extrude, revolve, loft, sweep, fillet, chamfer, shell, hole.

Все функции либо возвращают корректный cq.Shape, либо бросают GenerationError.
Подбор безопасных параметров (радиусов, толщин) выполняется на уровне ASTBuilder,
здесь же выполняется только детерминированное применение операции.
"""
import cadquery as cq
from typing import Any, Dict, Sequence

from ..exceptions import GenerationError
from .sketches import draw_sketch


def extrude(op: str, params: Dict[str, Any], distance: float) -> cq.Shape:
    """Экструзия эскиза `op`/`params` на расстояние `distance`."""
    wp = cq.Workplane(params.get("workplane", "XY"))
    wp = draw_sketch(wp, op, params)
    try:
        return wp.extrude(distance).val()
    except Exception as e:
        raise GenerationError(f"extrude failed: {e}") from e


def revolve(op: str, params: Dict[str, Any], angle: float = 360.0) -> cq.Shape:
    """Вращение эскиза вокруг нормали его плоскости (по умолчанию ось Z через начало)."""
    wp = cq.Workplane(params.get("workplane", "XY"))
    wp = draw_sketch(wp, op, params)
    try:
        return wp.revolve(angle).val()
    except Exception as e:
        raise GenerationError(f"revolve failed: {e}") from e


def loft(profiles: Sequence[Any], offsets: Sequence[float]) -> cq.Shape:
    """Лофт нескольких эскизов, расположенных на высотах `offsets` вдоль нормали XY.

    `profiles` - список пар (op, params) эскизов, все рисуются на плоскости XY.
    """
    if len(profiles) < 2:
        raise GenerationError("loft requires at least two profiles")
    if len(profiles) != len(offsets):
        raise GenerationError("loft profile/offset count mismatch")

    op0, params0 = profiles[0]
    wp = cq.Workplane("XY")
    wp = draw_sketch(wp, op0, params0)
    prev = offsets[0]
    for (op, params), z in zip(profiles[1:], offsets[1:]):
        gap = z - prev
        if gap <= 0.0:
            raise GenerationError("loft offsets must be strictly increasing")
        wp = wp.workplane(offset=gap)
        wp = draw_sketch(wp, op, params)
        prev = z
    try:
        return wp.loft().val()
    except Exception as e:
        raise GenerationError(f"loft failed: {e}") from e


def make_path_wire(path: Dict[str, Any]) -> cq.Wire:
    """Строит проволоку траектории для sweep по списку 3D-точек."""
    pts = [tuple(float(c) for c in p) for p in path["points"]]
    if len(pts) < 2:
        raise GenerationError("sweep path needs at least two points")
    method = path.get("method", "spline")
    wp = cq.Workplane(path.get("workplane", "XY"))
    try:
        if method == "polyline":
            return wp.polyline(pts).val()
        else:
            return wp.spline(pts).val()
    except Exception as e:
        raise GenerationError(f"path wire failed: {e}") from e


def sweep(profile_op: str, profile_params: Dict[str, Any], path: Dict[str, Any]) -> cq.Shape:
    """Протягивание профиля вдоль траектории (wire)."""
    wp = cq.Workplane(profile_params.get("workplane", "XY"))
    wp = draw_sketch(wp, profile_op, profile_params)
    path_wire = make_path_wire(path)
    try:
        return wp.sweep(path_wire).val()
    except Exception as e:
        raise GenerationError(f"sweep failed: {e}") from e


# Скругление/фаска всех рёбер сложного тела может быть очень медленной или
# «зависнуть» в OCCT, поэтому ограничиваемся детерминированным подмножеством —
# первыми 12 рёбрами в топологическом порядке. Топологический порядок stable,
# поэтому результат совпадает между исполнителем и сгенерированной программой.
_EDGE_LIMIT = 12


def fillet(solid: cq.Shape, radius: float) -> cq.Shape:
    """Скругление подмножества рёбер твёрдого тела через нативный API CadQuery."""
    try:
        return solid.fillet(radius, list(solid.edges())[:_EDGE_LIMIT])
    except Exception as e:
        raise GenerationError(f"fillet failed: {e}") from e


def chamfer(solid: cq.Shape, distance: float) -> cq.Shape:
    """Фаска подмножества рёбер твёрдого тела (симметричная)."""
    try:
        return solid.chamfer(distance, None, list(solid.edges())[:_EDGE_LIMIT])
    except Exception as e:
        raise GenerationError(f"chamfer failed: {e}") from e


def shell(solid: cq.Shape, thickness: float, face_rank: int = 0) -> cq.Shape:
    """Выдалбливание твёрдого тела с удалением грани с рангом `face_rank`.

    Порядок граней берётся из `list(shape.faces())` — топологический порядок,
    детерминированный для идентично построенного BRep, поэтому совпадает между
    исполнителем и сгенерированной программой.
    """
    faces = list(solid.faces())
    if not faces:
        raise GenerationError("shape has no faces for shell")
    face = faces[min(face_rank, len(faces) - 1)]
    try:
        return solid.shell([face], thickness)
    except Exception as e:
        raise GenerationError(f"shell failed: {e}") from e


def hole(solid: cq.Shape, position, radius: float, depth: float, axis=(0.0, 0.0, 1.0)) -> cq.Shape:
    """Сквозное отверстие: вычитание цилиндра. Ось задаётся вектором `axis`."""
    cylinder = cq.Solid.makeCylinder(radius, depth, cq.Vector(*position), cq.Vector(*axis))
    try:
        return solid.cut(cylinder)
    except Exception as e:
        raise GenerationError(f"hole cut failed: {e}") from e
