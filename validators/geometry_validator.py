"""Проверка корректности геометрии.

Критерии валидности:
  * shape не пустой и является единым твёрдым телом (Solid);
  * тело корректно по BRepCheck_Analyzer (при необходимости чинится ShapeFix_Shape);
  * объём конечен и лежит в [min_volume, max_volume];
  * диагональ ограничивающего параллелепипеда конечна и лежит в [min_bbox_diag, max_bbox_diag];
  * нет NaN/Inf в объёме и габаритах.
"""
import numpy as np
import cadquery as cq
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.ShapeFix import ShapeFix_Shape

from ..config import GeneratorConfig
from ..utils.geometry_utils import fast_bbox, unwrap_solid


class ValidationError(Exception):
    pass


class GeometryValidator:
    def __init__(self, config: GeneratorConfig):
        self.config = config

    def validate(self, shape: cq.Shape, is_root: bool = False):
        """Бросает ValidationError, если геометрия невалидна. Возвращает исправленный shape."""
        if shape is None:
            raise ValidationError("Shape is None")
        wrapped = getattr(shape, "wrapped", None)
        if wrapped is None or wrapped.IsNull():
            raise ValidationError("Shape is null")

        if is_root:
            shape = self._check_and_fix(shape)

        # булевы операции и массивы в CadQuery 2.7 возвращают Compound даже при
        # пересечении; распаковываем единственный Solid
        shape = unwrap_solid(shape)
        if shape.ShapeType() != "Solid":
            raise ValidationError(f"Shape is not a Solid (got {shape.ShapeType()})")

        # Убеждаемся, что тело состоит ровно из одного солида (нет разорванных частей).
        solids = list(shape.Solids())
        if len(solids) != 1:
            raise ValidationError(f"Expected single solid, got {len(solids)}")

        volume = float(shape.Volume())
        if not np.isfinite(volume):
            raise ValidationError(f"Volume is NaN/Inf: {volume}")
        if volume < self.config.min_volume:
            raise ValidationError(f"Volume too small: {volume}")
        if volume > self.config.max_volume:
            raise ValidationError(f"Volume too large: {volume}")

        xmin, xmax, ymin, ymax, zmin, zmax = fast_bbox(shape)
        xlen = xmax - xmin
        ylen = ymax - ymin
        zlen = zmax - zmin
        diag = np.sqrt(xlen ** 2 + ylen ** 2 + zlen ** 2)
        if not np.isfinite(diag):
            raise ValidationError(f"Bounding box diagonal is NaN/Inf: {diag}")
        if diag < self.config.min_bbox_diag:
            raise ValidationError(f"Bounding box diagonal too small: {diag}")
        if diag > self.config.max_bbox_diag:
            raise ValidationError(f"Bounding box diagonal too large: {diag}")

        return shape

    def _check_and_fix(self, shape: cq.Shape) -> cq.Shape:
        analyzer = BRepCheck_Analyzer(shape.wrapped)
        if analyzer.IsValid():
            return shape
        # Пытаемся исправить мелкие дефекты топологии.
        try:
            fixer = ShapeFix_Shape(shape.wrapped)
            fixer.Perform()
            fixed_wrapped = fixer.Shape()
            if fixed_wrapped.IsNull():
                raise ValidationError("Shape invalid and ShapeFix returned null")
            fixed = cq.Shape(fixed_wrapped)
            if fixed.isValid() and not fixed.wrapped.IsNull():
                return fixed
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"ShapeFix failed: {e}") from e
        raise ValidationError("Shape is invalid and could not be fixed")
