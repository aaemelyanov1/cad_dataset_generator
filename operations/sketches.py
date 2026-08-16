"""Работа с эскизами: рисование 2D-профилей на Workplane из параметров узла AST.

Параметры эскиза (params):
    workplane: str            - имя плоскости ("XY", "XZ", "YZ")
    center: (float, float)    - смещение центра эскиза в плоскости (по умолчанию (0, 0))
    а также специфичные параметры каждой фигуры (width, height, radius, ...).

Семейства эскизов:
    * базовые (замкнутый контур): rect, circle, ellipse, polygon, slot,
      polyline (выпуклый), spline (выпуклый);
    * roundrect  — скруглённый прямоугольник (rect + offset2D);
    * frame      — два вложенных контура (внешний + внутренний) — «прокладка»;
    * sector     — круговой сектор (polarLineTo + threePointArc);
    * arc_profile— прямоугольник с выгнутой стороной (threePointArc);
    * ellipse_arc— полуэллипс (ellipseArc + хорда);
    * bent       — «гнутый» профиль L/U (vLine/hLine);
    * mirrored   — симметричный профиль «из половины» (polyline + mirrorX).
"""
import math
import cadquery as cq
from typing import Any, Dict, List, Tuple


def _draw_basic(wp: cq.Workplane, op: str, params: Dict[str, Any]) -> cq.Workplane:
    """Рисует один базовый замкнутый контур (без обработки центра)."""
    if op == "rect":
        return wp.rect(params["width"], params["height"])
    if op == "circle":
        return wp.circle(params["radius"])
    if op == "ellipse":
        return wp.ellipse(params["x_radius"], params["y_radius"])
    if op == "polygon":
        return wp.polygon(int(params["n_sides"]), params["radius"])
    if op == "slot":
        return wp.slot2D(params["length"], params["width"], 0.0)
    raise ValueError(f"Unknown basic sketch op: {op}")


def draw_sketch(wp: cq.Workplane, op: str, params: Dict[str, Any]) -> cq.Workplane:
    """Рисует эскиз `op` на уже существующем workplane `wp`.

    Возвращает workplane с нанесённым профилем (wire). Учитывает параметр `center`.
    """
    center = params.get("center", (0.0, 0.0))
    cx, cy = float(center[0]), float(center[1])
    if op in ("polyline", "spline"):
        # moveTo не влияет на polyline/spline (координаты абсолютные),
        # поэтому смещение центра добавляем к каждой точке напрямую
        pts = [(float(p[0]) + cx, float(p[1]) + cy) for p in params["points"]]
        if op == "polyline":
            return wp.polyline(pts).close()
        return wp.spline(pts).close()
    if op == "mirrored":
        # mirrorX зеркалит относительно оси X через начало координат
        if cx != 0.0 or cy != 0.0:
            raise ValueError("mirrored требует center=(0, 0)")
        return wp.polyline(params["points"]).mirrorX()
    if op == "ellipse_arc":
        # эллиптическая дуга строится от точки (-xr, 0) относительно начала
        if cx != 0.0 or cy != 0.0:
            raise ValueError("ellipse_arc требует center=(0, 0)")
        xr, yr = float(params["x_radius"]), float(params["y_radius"])
        return (wp.moveTo(-xr, 0.0).ellipseArc(xr, yr, 0.0, 180.0)
                .lineTo(-xr, 0.0).close())
    if op == "arc_profile":
        # прямоугольник с одной выгнутой стороной; координаты абсолютные,
        # поэтому центр запекаем в каждую точку
        w, h, bow = float(params["width"]), float(params["height"]), float(params.get("bow", 0.0))
        return (wp.moveTo(-w / 2 + cx, -h / 2 + cy)
                .lineTo(w / 2 + cx, -h / 2 + cy)
                .threePointArc((w / 2 + bow + cx, cy), (w / 2 + cx, h / 2 + cy))
                .lineTo(-w / 2 + cx, h / 2 + cy)
                .close())
    if cx != 0.0 or cy != 0.0:
        wp = wp.moveTo(cx, cy)

    if op == "frame":
        # внешний и внутренний контуры рисуются в одной точке центра;
        # базовые фигуры не двигают курсор, поэтому оба центрируются одинаково
        wp = _draw_basic(wp, params["frame_op"], params["outer"])
        return _draw_basic(wp, params["frame_op"], params["inner"])
    if op == "roundrect":
        return wp.rect(params["width"], params["height"]).offset2D(params["radius"])
    if op == "sector":
        r = float(params["radius"])
        ang = float(params["angle"])
        half = math.radians(ang / 2.0)
        full = math.radians(ang)
        mid = (cx + r * math.cos(half), cy + r * math.sin(half))
        end = (cx + r * math.cos(full), cy + r * math.sin(full))
        return wp.polarLineTo(r, 0.0).threePointArc(mid, end).close()
    if op == "bent":
        shape = params.get("shape", "L")
        a, b, t = float(params["a"]), float(params["b"]), float(params["t"])
        if shape == "L":
            return (wp.hLine(a).vLine(t).hLine(-(a - b)).vLine(b)
                    .hLine(-t).vLine(-b).close())
        if shape == "U":
            return (wp.hLine(a).vLine(b).hLine(-t).vLine(-(b - 2 * t))
                    .hLine(-(a - 2 * t)).vLine(b - 2 * t).hLine(-t).close())
        raise ValueError(f"Unknown bent shape: {shape}")

    if op == "rect":
        return wp.rect(params["width"], params["height"])
    elif op == "circle":
        return wp.circle(params["radius"])
    elif op == "ellipse":
        return wp.ellipse(params["x_radius"], params["y_radius"])
    elif op == "polygon":
        return wp.polygon(int(params["n_sides"]), params["radius"])
    elif op == "slot":
        return wp.slot2D(params["length"], params["width"], 0.0)
    raise ValueError(f"Unknown sketch op: {op}")


def sketch_to_workplane(op: str, params: Dict[str, Any]) -> cq.Workplane:
    """Создаёт новый workplane и рисует на нём эскиз `op`."""
    wp_name = params.get("workplane", "XY")
    wp = cq.Workplane(wp_name)
    return draw_sketch(wp, op, params)


def make_rect_sketch(width: float, height: float, workplane: str = "XY") -> cq.Shape:
    return sketch_to_workplane("rect", {"width": width, "height": height, "workplane": workplane}).val()


def make_circle_sketch(radius: float, workplane: str = "XY") -> cq.Shape:
    return sketch_to_workplane("circle", {"radius": radius, "workplane": workplane}).val()


def make_ellipse_sketch(x_radius: float, y_radius: float, workplane: str = "XY") -> cq.Shape:
    return sketch_to_workplane(
        "ellipse", {"x_radius": x_radius, "y_radius": y_radius, "workplane": workplane}
    ).val()


def make_polygon_sketch(n_sides: int, radius: float, workplane: str = "XY") -> cq.Shape:
    return sketch_to_workplane(
        "polygon", {"n_sides": n_sides, "radius": radius, "workplane": workplane}
    ).val()


def make_slot_sketch(length: float, width: float, workplane: str = "XY") -> cq.Shape:
    return sketch_to_workplane(
        "slot", {"length": length, "width": width, "workplane": workplane}
    ).val()


def make_polyline_sketch(points: List[Tuple[float, float]], workplane: str = "XY") -> cq.Shape:
    return sketch_to_workplane(
        "polyline", {"points": points, "workplane": workplane}
    ).val()


def make_spline_sketch(points: List[Tuple[float, float]], workplane: str = "XY") -> cq.Shape:
    return sketch_to_workplane(
        "spline", {"points": points, "workplane": workplane}
    ).val()
