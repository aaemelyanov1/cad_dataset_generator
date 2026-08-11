"""Булевы операции над Shape. В CadQuery 2.7 у cq.Shape нет метода union, используется fuse."""
import cadquery as cq

from ..exceptions import GenerationError


def union(a: cq.Shape, b: cq.Shape) -> cq.Shape:
    try:
        return a.fuse(b)
    except Exception as e:
        raise GenerationError(f"union failed: {e}") from e


def cut(a: cq.Shape, b: cq.Shape) -> cq.Shape:
    try:
        return a.cut(b)
    except Exception as e:
        raise GenerationError(f"cut failed: {e}") from e


def intersect(a: cq.Shape, b: cq.Shape) -> cq.Shape:
    try:
        return a.intersect(b)
    except Exception as e:
        raise GenerationError(f"intersect failed: {e}") from e
