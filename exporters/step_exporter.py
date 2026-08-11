# ==============================
# exporters/step_exporter.py
# ==============================
import cadquery as cq
from pathlib import Path

class StepExporter:
    @staticmethod
    def export(shape: cq.Shape, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        cq.exporters.export(shape, str(path))