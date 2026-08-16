import numpy as np
import pyrender
import trimesh
import cadquery as cq
from pathlib import Path
from PIL import Image

class RenderExporter:
    @staticmethod
    def export(shape: cq.Shape, path: Path, resolution=(512, 512)):
        vertices, faces = shape.tessellate(0.1)
        if not vertices or not faces:
            raise ValueError("empty tessellation for render")
        verts_array = np.array([[v.x, v.y, v.z] for v in vertices], dtype=np.float64)
        faces_array = np.array(faces, dtype=np.int64).reshape(-1, 3)
        tri_mesh = trimesh.Trimesh(vertices=verts_array, faces=faces_array)
        # Кадрирование по bbox: камера ставится на расстоянии ~3×max extent и
        # смотрит на центр модели (раньше она стояла в origin с identity-pose,
        # из-за чего кадр был белым/пустым для тел у начала координат).
        bmin, bmax = tri_mesh.bounds
        center = (bmin + bmax) * 0.5
        extent = bmax - bmin
        max_dim = float(np.max(extent)) if extent.size else 1.0
        max_dim = max(max_dim, 1e-3)
        distance = max_dim * 4.5
        eye = center + np.array([0.8, -0.9, 1.2]) * distance

        forward = center - eye
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        cam_up = np.cross(right, forward)

        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = cam_up
        pose[:3, 2] = -forward  # камера смотрит вдоль -Z
        pose[:3, 3] = eye

        scene = pyrender.Scene()
        mesh_py = pyrender.Mesh.from_trimesh(tri_mesh)
        scene.add(mesh_py)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=pose)
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
        scene.add(light, pose=np.eye(4))

        r = pyrender.OffscreenRenderer(resolution[0], resolution[1])
        color, _ = r.render(scene)
        img = Image.fromarray(color)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path))
        r.delete()
