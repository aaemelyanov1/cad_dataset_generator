"""Работа с эскизами: рисование 2D-профилей на Workplane из параметров узла AST.

Параметры эскиза (params):
    workplane: str            - имя плоскости ("XY", "XZ", "YZ")
    center: (float, float)    - смещение центра эскиза в плоскости (по умолчанию (0, 0))
    а также специфичные параметры каждой фигуры (width, height, radius, ...).
"""
import cadquery as cq
from typing import Any, Dict, List, Tuple


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
    if cx != 0.0 or cy != 0.0:
        wp = wp.moveTo(cx, cy)

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
