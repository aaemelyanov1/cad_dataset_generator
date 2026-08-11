# ==============================
# exporters/stl_exporter.py
# ==============================
import cadquery as cq
from pathlib import Path

class StlExporter:
    @staticmethod
    def export(shape: cq.Shape, path: Path, tolerance=0.001):
        path.parent.mkdir(parents=True, exist_ok=True)
        cq.exporters.export(shape, str(path), exportType='STL', tolerance=tolerance)