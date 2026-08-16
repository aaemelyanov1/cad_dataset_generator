"""Тесты операций: каждый тип операции возвращает валидный непустой Shape."""
import numpy as np
import pytest

from cad_dataset_generator.operations import (
    make_box, make_cylinder, make_sphere, make_cone, make_wedge, make_torus,
    extrude, revolve, twist_extrude, loft, sweep,
    fillet, chamfer, shell, hole, split,
    translate, rotate, mirror,
    rectangular_array, polar_array, scatter,
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


def test_twist_extrude():
    rect = {"workplane": "XY", "center": (0.0, 0.0), "width": 3.0, "height": 3.0}
    t = twist_extrude("rect", rect, 4.0, 30.0)
    solid = unwrap_solid(t)
    assert _volume(solid) > 0
    assert len(list(solid.Solids())) == 1


def test_split_reduces_volume():
    base = make_box(6, 4, 3)
    base_vol = _volume(base)
    s = unwrap_solid(split(base, "Z", 0.9))
    assert len(list(s.Solids())) == 1
    assert 0 < _volume(s) < base_vol


@pytest.mark.parametrize("kind,pos,depth,extra", [
    ("through", (0.0, 0.0, -1.0), 8.0, {}),
    ("blind", (0.0, 0.0, 3.0), 2.0, {}),
    ("cbore", (0.0, 0.0, 3.0), 2.0, {"cbo_radius": 1.5, "cbo_depth": 1.2}),
    ("csk", (0.0, 0.0, 3.0), 2.0, {"csk_radius": 1.8, "csk_depth": 1.0}),
])
def test_hole_kinds_reduce_volume(kind, pos, depth, extra):
    base = make_box(6, 6, 6)
    base_vol = _volume(base)
    h = unwrap_solid(hole(base, pos, 0.7, depth, kind=kind, **extra))
    assert len(list(h.Solids())) == 1
    assert 0 < _volume(h) < base_vol


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
    s = shell(base, 0.2, {"kind": "direction", "direction": (0.0, 0.0, 1.0)})
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
    shapes = [
        rectangular_array(base, 3, 2, 1.2, 1.2),
        polar_array(base, 5, radius=0.8),
        scatter(base, [(0.0, 0.0), (0.8, 0.0), (0.0, 0.8)]),
    ]
    for shape in shapes:
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
