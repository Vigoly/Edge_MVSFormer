import open3d as o3d
import numpy as np
def compute_mesh_area(mesh):
    area = 0.0
    for triangle in mesh.triangles:
        p0 = mesh.vertices[triangle[0]]
        p1 = mesh.vertices[triangle[1]]
        p2 = mesh.vertices[triangle[2]]
        area += 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
    return abs(area)

# 加载网格
mesh = o3d.io.read_triangle_mesh("all_torch_mesh_danyepian.ply")

# 计算面积
mesh_area = compute_mesh_area(mesh)
print(f"The area of the mesh is: {mesh_area} square units")