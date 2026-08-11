"""Тесты валидатора геометрии."""
import numpy as np
import pytest
import cadquery as cq

from cad_dataset_generator.config import GeneratorConfig
from cad_dataset_generator.operations import make_box
from cad_dataset_generator.utils.geometry_utils import unwrap_solid
from cad_dataset_generator.validators.geometry_validator import GeometryValidator, ValidationError


@pytest.fixture
def validator():
    return GeometryValidator(GeneratorConfig())


def test_accepts_valid_box(validator):
    result = validator.validate(make_box(2, 2, 2), is_root=True)
    assert result is not None
    assert result.ShapeType() == "Solid"


def test_rejects_none(validator):
    with pytest.raises(ValidationError):
        validator.validate(None)


def test_rejects_too_small_volume():
    config = GeneratorConfig(min_volume=100.0)
    v = GeometryValidator(config)
    with pytest.raises(ValidationError):
        v.validate(make_box(1, 1, 1), is_root=True)


def test_rejects_too_small_bbox():
    config = GeneratorConfig(min_bbox_diag=100.0)
    v = GeometryValidator(config)
    with pytest.raises(ValidationError):
        v.validate(make_box(1, 1, 1), is_root=True)


def test_rejects_compound_of_many_solids(validator):
    parts = []
    for i in range(3):
        box = make_box(1, 1, 1)
        from cadquery import Vector
        box = box.translate((Vector(0, 0, i * 2.0)))
        parts.append(box)
    compound = cq.Compound.makeCompound(parts)
    with pytest.raises(ValidationError):
        validator.validate(compound, is_root=False)
