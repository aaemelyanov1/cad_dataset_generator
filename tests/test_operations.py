"""Тесты операций: каждый тип операции возвращает валидный непустой Shape."""
import numpy as np
import pytest

from cad_dataset_generator.operations import (
    make_box, make_cylinder, make_sphere, make_cone, make_wedge, make_torus,
    extrude, revolve, loft, sweep,
    fillet, chamfer, shell, hole,
    translate, rotate, mirror,
    rectangular_array, polar_array,
    union, cut, intersect,
)
from cad_dataset_generator.utils.geometry_utils import unwrap_solid


def _volume(shape):
    return float(shape.Volume())


def test_primitives_positive_volume():
    for shape in [
        make_box(2, 3, 4),
        make_cylinder(5, 1),
        make_sphere(2),
        make_cone(4, 1.5, 0.5),
        make_wedge(3, 4, 5, 0, 0, 2, 4),
        make_torus(3, 0.5),
    ]:
        assert _volume(shape) > 0


def test_extrude_revolve():
    rect = {"workplane": "XY", "center": (0.0, 0.0), "width": 2.0, "height": 3.0}
    assert _volume(extrude("rect", rect, 4.0)) > 0
    circle = {"workplane": "XY", "center": (5.0, 0.0), "radius": 1.0}
    assert _volume(revolve("circle", circle, 360.0)) > 0


@pytest.mark.parametrize("op", ["polyline", "spline"])
def test_extrude_polyline_spline(op):
    sk = {"workplane": "XY", "center": (0.0, 0.0),
          "points": [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]}
    assert _volume(extrude(op, sk, 3.0)) > 0


@pytest.mark.parametrize("op", ["polyline", "spline"])
def test_revolve_polyline_spline(op):
    sk = {"workplane": "XY", "center": (3.0, 0.0),
          "points": [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]}
    assert _volume(revolve(op, sk, 360.0)) > 0


def test_loft():
    profiles = [
        ("rect", {"workplane": "XY", "center": (0.0, 0.0), "width": 4.0, "height": 4.0}),
        ("rect", {"workplane": "XY", "center": (0.0, 0.0), "width": 2.0, "height": 2.0}),
    ]
    assert _volume(loft(profiles, [0.0, 5.0])) > 0


def test_sweep():
    path = {"workplane": "XY", "points": [(0, 0, 0), (2, 0, 0.5), (2, 2, 1)], "method": "polyline"}
    assert _volume(sweep("circle", {"workplane": "XY", "center": (0.0, 0.0), "radius": 0.5}, path)) > 0


def test_fillet_chamfer_reduce_volume():
    base = make_box(4, 4, 4)
    r = fillet(base, 0.5)
    assert 0 < _volume(r) < _volume(base)
    c = chamfer(base, 0.5)
    assert 0 < _volume(c) < _volume(base)


def test_shell_reduces_volume():
    base = make_box(4, 4, 4)
    s = shell(base, 0.2, face_rank=0)
    assert 0 < _volume(s) < _volume(base)


def test_hole_reduces_volume():
    base = make_box(4, 4, 4)
    h = hole(base, (0.0, 0.0, -1.0), 0.5, 6.0)
    assert 0 < _volume(h) < _volume(base)


def test_transforms_preserve_volume():
    base = make_box(2, 2, 2)
    v = _volume(base)
    assert abs(_volume(translate(base, (3, 4, 5))) - v) < 1e-6
    assert abs(_volume(rotate(base, "Z", 45.0)) - v) < 1e-6
    assert abs(_volume(mirror(base, "XY")) - v) < 1e-6


def test_patterns_single_solid():
    base = make_box(2, 2, 2)
    for shape in [rectangular_array(base, 3, 2, 1.2, 1.2), polar_array(base, 5)]:
        solid = unwrap_solid(shape)
        assert len(list(solid.Solids())) == 1
        assert _volume(solid) > 0


def test_booleans():
    a = make_box(2, 2, 2)
    b = translate(make_box(2, 2, 2), (1, 0, 0))
    u = unwrap_solid(union(a, b))
    assert _volume(u) > _volume(a)
    c = unwrap_solid(cut(a, b))
    assert _volume(c) < _volume(a)
    i = unwrap_solid(intersect(a, b))
    assert 0 < _volume(i) < _volume(a)
