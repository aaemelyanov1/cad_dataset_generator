import cadquery as cq
import trimesh
import pyrender
import numpy as np
from pathlib import Path
from PIL import Image

class RenderExporter:
    @staticmethod
    def export(shape: cq.Shape, path: Path, resolution=(512, 512)):
        vertices, faces = shape.tessellate(0.1)
        verts_array = np.array([[v.x, v.y, v.z] for v in vertices], dtype=np.float64)
        faces_array = np.array(faces, dtype=np.int64).reshape(-1, 3)
        tri_mesh = trimesh.Trimesh(vertices=verts_array, faces=faces_array)

        scene = pyrender.Scene()
        mesh_py = pyrender.Mesh.from_trimesh(tri_mesh)
        scene.add(mesh_py)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=np.eye(4))
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
        scene.add(light, pose=np.eye(4))

        r = pyrender.OffscreenRenderer(resolution[0], resolution[1])
        color, _ = r.render(scene)
        img = Image.fromarray(color)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path))
        r.delete()