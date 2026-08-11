import cadquery as cq
import trimesh
import numpy as np
from pathlib import Path

class MeshExporter:
    @staticmethod
    def export(shape: cq.Shape, path: Path, tolerance=0.01):
        vertices, faces = shape.tessellate(tolerance)
        verts_array = np.array([[v.x, v.y, v.z] for v in vertices], dtype=np.float64)
        faces_array = np.array(faces, dtype=np.int64).reshape(-1, 3)
        tri_mesh = trimesh.Trimesh(vertices=verts_array, faces=faces_array)
        path.parent.mkdir(parents=True, exist_ok=True)
        tri_mesh.export(str(path))