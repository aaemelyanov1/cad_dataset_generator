import numpy as np
import trimesh
import cadquery as cq
from pathlib import Path

class PointCloudExporter:
    @staticmethod
    def export(shape: cq.Shape, path: Path, n_samples=4096):
        vertices, faces = shape.tessellate(0.1)
        verts_array = np.array([[v.x, v.y, v.z] for v in vertices], dtype=np.float64)
        faces_array = np.array(faces, dtype=np.int64).reshape(-1, 3)
        tri_mesh = trimesh.Trimesh(vertices=verts_array, faces=faces_array)
        points, _ = trimesh.sample.sample_surface(tri_mesh, n_samples)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), points)