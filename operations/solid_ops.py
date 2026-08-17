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


def twist_extrude(op: str, params: Dict[str, Any], distance: float, angle: float) -> cq.Shape:
    """Экструзия эскиза с одновременным поворотом на `angle` градусов.

    Воспроизводит нативный вызов twistExtrude (clean=False, чтобы не наткнуться
    на баг Shape.clean(), ломающий грани).
    """
    wp = cq.Workplane(params.get("workplane", "XY"))
    wp = draw_sketch(wp, op, params)
    try:
        return wp.twistExtrude(distance, angle, clean=False).val()
    except Exception as e:
        raise GenerationError(f"twist_extrude failed: {e}") from e


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


# --------------------------------------------------------------------------- #
# Выбор рёбер/граней через нативные Selectors CadQuery.
#
# Семантическая выборка небольшого подмножества рёбер (1-2) вместо «всех первых
# N в топологическом порядке» устраняет главный источник зависаний fillet/chamfer
# в OCCT на сложных телах и делает модификации «человечными» (скруглить только
# рёбра у точки, фаску — по направлению).
#
# Selection описывается словарём:
#   {"kind": "nearest",   "point": (x, y, z)}          - рёбра/грани у точки
#   {"kind": "direction", "direction": (dx, dy, dz)}   - рёбра вдоль / грани
#                                                        перпендикулярно вектору
# --------------------------------------------------------------------------- #

def select_edges(solid: cq.Shape, selection: Dict[str, Any], limit: int = 2):
    kind = selection.get("kind", "nearest")
    if kind == "nearest":
        pt = tuple(float(c) for c in selection["point"])
        return cq.NearestToPointSelector(pt).filter(solid.Edges())[:limit]
    if kind == "direction":
        d = tuple(float(c) for c in selection["direction"])
        return cq.DirectionSelector(cq.Vector(*d)).filter(solid.Edges())[:limit]
    raise GenerationError(f"Unknown edge selection kind: {kind}")


def select_face(solid: cq.Shape, selection: Dict[str, Any]):
    kind = selection.get("kind", "direction")
    if kind == "direction":
        d = tuple(float(c) for c in selection["direction"])
        faces = cq.DirectionSelector(cq.Vector(*d)).filter(solid.Faces())
        if not faces:
            raise GenerationError("no face matches DirectionSelector")
        return faces[0]
    if kind == "nearest":
        pt = tuple(float(c) for c in selection["point"])
        faces = cq.NearestToPointSelector(pt).filter(solid.Faces())
        if not faces:
            raise GenerationError("no face matches NearestToPointSelector")
        return faces[0]
    raise GenerationError(f"Unknown face selection kind: {kind}")


def fillet(solid: cq.Shape, radius: float, selection: Dict[str, Any] = None) -> cq.Shape:
    """Скругление подмножества рёбер (по умолчанию — рёбра, ближайшие к началу)."""
    sel = selection or {"kind": "nearest", "point": (0.0, 0.0, 0.0)}
    try:
        edges = select_edges(solid, sel, limit=2)
        if not edges:
            raise GenerationError("no edges selected for fillet")
        return solid.fillet(radius, edges)
    except GenerationError:
        raise
    except Exception as e:
        raise GenerationError(f"fillet failed: {e}") from e


def chamfer(solid: cq.Shape, distance: float, selection: Dict[str, Any] = None) -> cq.Shape:
    """Фаска подмножества рёбер (по умолчанию — рёбра, ближайшие к началу)."""
    sel = selection or {"kind": "nearest", "point": (0.0, 0.0, 0.0)}
    try:
        edges = select_edges(solid, sel, limit=2)
        if not edges:
            raise GenerationError("no edges selected for chamfer")
        return solid.chamfer(distance, None, edges)
    except GenerationError:
        raise
    except Exception as e:
        raise GenerationError(f"chamfer failed: {e}") from e


def shell(solid: cq.Shape, thickness: float, selection: Dict[str, Any] = None) -> cq.Shape:
    """Выдалбливание твёрдого тела с удалением грани, выбранной селектором.

    По умолчанию удаляется грань, нормаль которой совпадает с +Z.
    """
    sel = selection or {"kind": "direction", "direction": (0.0, 0.0, 1.0)}
    try:
        face = select_face(solid, sel)
        return solid.shell([face], thickness)
    except GenerationError:
        raise
    except Exception as e:
        raise GenerationError(f"shell failed: {e}") from e


def hole(solid: cq.Shape, position, radius: float, depth: float,
         kind: str = "through",
         cbo_radius: float = None, cbo_depth: float = None,
         csk_radius: float = None, csk_depth: float = None,
         axis=(0.0, 0.0, 1.0)) -> cq.Shape:
    """Отверстие одного из видов: through / blind / cbore / csk.

    Всё выполняется нативными вырезами цилиндра/конуса вдоль +Z:
      * through — сквозное отверстие (цилиндр длиной `depth` из-под низа);
      * blind   — глухое: цилиндр от zmax-depth до zmax (не доходит до низа);
      * cbore   — цековка: широкий мелкий цилиндр у верха + узкий длинный;
      * csk     — зенковка: конус у верха + узкий цилиндр ниже.
    """
    try:
        if kind == "through":
            cyl = cq.Solid.makeCylinder(radius, depth, cq.Vector(*position), cq.Vector(*axis))
            return solid.cut(cyl)
        if kind == "blind":
            # глухое: цилиндр от (position.z - depth) до position.z — карман,
            # открытый на верхней грани (position.z = zmax), не доходящий до низа
            z_top = float(position[2])
            cyl = cq.Solid.makeCylinder(radius, depth,
                                        cq.Vector(position[0], position[1], z_top - depth),
                                        cq.Vector(*axis))
            return solid.cut(cyl)
        if kind == "cbore":
            if cbo_radius is None or cbo_depth is None:
                raise GenerationError("cbore requires cbo_radius/cbo_depth")
            z_top = float(position[2])
            head = cq.Solid.makeCylinder(cbo_radius, cbo_depth,
                                         cq.Vector(position[0], position[1], z_top - cbo_depth),
                                         cq.Vector(*axis))
            bore = cq.Solid.makeCylinder(radius, depth,
                                         cq.Vector(position[0], position[1], z_top - cbo_depth),
                                         cq.Vector(*axis))
            return solid.cut(head).cut(bore)
        if kind == "csk":
            if csk_radius is None or csk_depth is None:
                raise GenerationError("csk requires csk_radius/csk_depth")
            z_top = float(position[2])
            cone = cq.Solid.makeCone(radius, csk_radius, csk_depth,
                                     cq.Vector(position[0], position[1], z_top - csk_depth),
                                     cq.Vector(*axis))
            bore = cq.Solid.makeCylinder(radius, depth,
                                         cq.Vector(position[0], position[1], z_top - csk_depth),
                                         cq.Vector(*axis))
            return solid.cut(cone).cut(bore)
        raise GenerationError(f"Unknown hole kind: {kind}")
    except GenerationError:
        raise
    except Exception as e:
        raise GenerationError(f"hole cut failed: {e}") from e


# детерминированный подбор плоскости: сначала в середине bbox (обычно достаточно),
# затем два малых смещения для граничных случаев (плоскость на ребре/вершине).
# Больше попыток не делаем: более экзотичные расклады — валидатор/бэктрекинг.
_SPLIT_OFFSETS = (0.0, 1e-4, -1e-4)


def _split_reference(solid: cq.Shape, axis: str):
    """Возвращает (ref_solid, plane_center, extent_по_axis).

    Для одиночного Solid центр плоскости — `solid.Center()` (в точности старое
    поведение). Для Compound — центр bbox **наибольшего** solid'а: `shape.Center()`
    у составных тел (TopoDS-pivot) давал плоскости мимо середины и
    «split produced 1 halves».
    """
    solids = list(solid.Solids())
    if not solids:
        raise GenerationError("split: shape has no solids")
    ref = max(solids, key=lambda s: float(s.Volume()))
    if len(solids) == 1:
        center = solid.Center()
    else:
        bb = ref.BoundingBox()
        center = cq.Vector((bb.xmin + bb.xmax) / 2.0,
                           (bb.ymin + bb.ymax) / 2.0,
                           (bb.zmin + bb.zmax) / 2.0)
    bb = ref.BoundingBox()
    low = {"X": bb.xmin, "Y": bb.ymin, "Z": bb.zmin}[axis]
    high = {"X": bb.xmax, "Y": bb.ymax, "Z": bb.zmax}[axis]
    extent = float(high - low)
    if extent <= 0.0:
        raise GenerationError("split: zero extent along axis")
    return ref, center, extent


def resolve_split(solid: cq.Shape, axis: str, gap: float):
    """Детерминированный разрез с подбором плоскости.

    Пробует микросдвиги плоскости вдоль `axis` (0, ±1e-4)×extent, берёт первый,
    дающий >= 2 частей, и возвращает `(plane_offset_fraction, parts_count,
    fused_shape)`. Число частей фиксируется, чтобы code-генератор воспроизвёл
    тот же fuse без циклов в программе. Бросает GenerationError.
    """
    axis_vec = {"X": cq.Vector(1, 0, 0),
                "Y": cq.Vector(0, 1, 0),
                "Z": cq.Vector(0, 0, 1)}.get(axis)
    if axis_vec is None:
        raise GenerationError(f"Unknown split axis: {axis}")
    _, center, extent = _split_reference(solid, axis)
    shift = axis_vec * (-float(gap))
    for off in _SPLIT_OFFSETS:
        plane = cq.Face.makePlane(1e6, 1e6,
                                  center + axis_vec * (off * extent), axis_vec)
        try:
            parts = list(solid.split(plane).Solids())
        except Exception:
            # OCCT может вернуть Null TopoDS_Shape (или бросить) при
            # вырожденном разрезе (плоскость по грани/вершине, особенно на
            # малых масштабах, где линейная точность не видна) — пробуем
            # следующий микросдвиг плоскости
            continue
        if len(parts) < 2:
            continue
        parts.sort(key=lambda s: (float(s.Center().x),
                                  float(s.Center().y),
                                  float(s.Center().z)))
        fused = parts[0]
        for p in parts[1:]:
            fused = fused.fuse(p.translate(shift))
        return off, len(parts), fused
    raise GenerationError("split produced <2 halves for all fallback planes")


def split(solid: cq.Shape, axis: str, gap: float, plane_offset: float = None,
          parts_count: int = None) -> cq.Shape:
    """Разрез тела плоскостью со сдвигом части половин.

    Плоскость перпендикулярна `axis`; опорный центр — `solid.Center()` для
    одиночного Solid и середина bbox наибольшего solid'а для Compound
    (см. `_split_reference`). Если `plane_offset` не задан — плоскость
    подбирается детерминированно (`resolve_split`); иначе используется переданный
    (тот, что ASTBuilder зафиксировал в параметрах узла). Все части сортируются по
    координате вдоль `axis`; часть с минимальной координатой остаётся на месте,
    остальные смещаются на `gap` к ней (перекрытие), затем всё fuse-ится в единый
    Solid — «ступенчатый» разрез тела. При заданном `parts_count` обязательный
    процент совпадения числа частей (программа воспроизводит итог без циклов).
    """
    if plane_offset is None:
        try:
            _, _, fused = resolve_split(solid, axis, float(gap))
            return fused
        except GenerationError:
            raise
        except Exception as e:
            raise GenerationError(f"split failed: {e}") from e
    axis_vec = {"X": cq.Vector(1, 0, 0),
                "Y": cq.Vector(0, 1, 0),
                "Z": cq.Vector(0, 0, 1)}[axis]
    try:
        _, center, extent = _split_reference(solid, axis)
        plane = cq.Face.makePlane(1e6, 1e6,
                                  center + axis_vec * (plane_offset * extent), axis_vec)
        parts = list(solid.split(plane).Solids())
        if len(parts) < 2:
            raise GenerationError(f"split produced {len(parts)} halves")
        if parts_count is not None and len(parts) != parts_count:
            raise GenerationError(f"split produced {len(parts)} parts, expected {parts_count}")
        parts.sort(key=lambda s: (float(s.Center().x),
                                  float(s.Center().y),
                                  float(s.Center().z)))
        shift = axis_vec * (-float(gap))
        fused = parts[0]
        for p in parts[1:]:
            fused = fused.fuse(p.translate(shift))
        return fused
    except GenerationError:
        raise
    except Exception as e:
        raise GenerationError(f"split failed: {e}") from e
