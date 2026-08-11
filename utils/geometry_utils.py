# ==============================
# utils/geometry_utils.py
# ==============================
import cadquery as cq
from cadquery.occ_impl.geom import BoundBox


def fast_bbox(shape: cq.Shape):
    """Быстрый ограничивающий параллелепипед (без точного optimal-расчёта).

    BRepBndLib.AddOptimal (режим по умолчанию в cq.Shape.BoundingBox) очень
    дорогой, а для эвристик генератора достаточно обычного bbox, построенного
    по треугольной сетке. ВНИМАНИЕ: результат зависит от текущей триангуляции
    shape (изменяется после tessellate/экспорта), поэтому не используйте эту
    функцию для метаданных — только для эвристик билдера/валидатора.
    """
    bb = BoundBox._fromTopoDS(shape.wrapped, optimal=False)
    return (bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax)


def get_bounding_box(shape: cq.Shape):
    return fast_bbox(shape)


def unwrap_solid(shape: cq.Shape) -> cq.Shape:
    """Возвращает единственный Solid из Compound/CompSolid (если он там один).

    CadQuery 2.7 возвращает Compound даже при объединении пересекающихся тел;
    чтобы нижележащие операции (fillet, chamfer, shell) работали с Solid,
    распаковываем Compound с одним солидом.
    """
    st = getattr(shape, "ShapeType", lambda: None)()
    if st in ("Compound", "CompSolid"):
        solids = list(shape.Solids())
        if len(solids) == 1:
            return solids[0]
    return shape


def get_volume(shape: cq.Shape):
    return shape.Volume()
