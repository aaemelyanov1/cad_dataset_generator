"""Примитивные твёрдые тела."""
import cadquery as cq

from ..exceptions import GenerationError


def make_box(length: float, width: float, height: float) -> cq.Shape:
    try:
        return cq.Workplane("XY").box(length, width, height).val()
    except Exception as e:
        raise GenerationError(f"box failed: {e}") from e


def make_cylinder(height: float, radius: float) -> cq.Shape:
    try:
        return cq.Workplane("XY").cylinder(height, radius).val()
    except Exception as e:
        raise GenerationError(f"cylinder failed: {e}") from e


def make_sphere(radius: float) -> cq.Shape:
    try:
        return cq.Workplane("XY").sphere(radius).val()
    except Exception as e:
        raise GenerationError(f"sphere failed: {e}") from e


def make_cone(height: float, radius1: float, radius2: float) -> cq.Shape:
    """Конус: radius1 - радиус основания, radius2 - радиус вершины."""
    try:
        return cq.Solid.makeCone(radius1, radius2, height)
    except Exception as e:
        raise GenerationError(f"cone failed: {e}") from e


def make_wedge(dx: float, dy: float, dz: float,
               xmin: float, zmin: float, xmax: float, zmax: float) -> cq.Shape:
    try:
        return cq.Workplane("XY").wedge(dx, dy, dz, xmin, zmin, xmax, zmax).val()
    except Exception as e:
        raise GenerationError(f"wedge failed: {e}") from e


def make_torus(radius1: float, radius2: float) -> cq.Shape:
    """Тор: radius1 - большой радиус, radius2 - малый радиус."""
    try:
        return cq.Solid.makeTorus(radius1, radius2)
    except Exception as e:
        raise GenerationError(f"torus failed: {e}") from e
