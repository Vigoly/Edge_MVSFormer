import numpy as np
import open3d as o3d

def read_ply(file_path):
    return o3d.io.read_point_cloud(file_path)

def read_mesh(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_vertex_normals()  # 计算顶点法线
    return mesh

def get_edge_points_from_mesh(mesh, angle_threshold=np.pi / 6):
    mesh.compute_vertex_normals()
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    edges = {}

    for tri in triangles:
        for i in range(3):
            edge = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
            if edge not in edges:
                edges[edge] = []
            edges[edge].append(tri)

    edge_points = []

    for edge, faces in edges.items():
        if len(faces) == 1:
            edge_points.append(vertices[edge[0]])
            edge_points.append(vertices[edge[1]])

    edge_points = np.unique(edge_points, axis=0)
    return edge_points

def filter_dense_points(dense_points, edge_points, threshold):
    if edge_points.size == 0:
        print("No edge points to filter against.")
        return dense_points

    edge_pcd = o3d.geometry.PointCloud()
    edge_pcd.points = o3d.utility.Vector3dVector(edge_points)

    dense_pcd = o3d.geometry.PointCloud()
    dense_pcd.points = o3d.utility.Vector3dVector(dense_points)

    kdtree = o3d.geometry.KDTreeFlann(edge_pcd)
    filtered_points = []

    for point in dense_points:
        [_, idx, _] = kdtree.search_knn_vector_3d(point, 1)
        distance = np.linalg.norm(point - edge_points[idx[0]])
        if distance < threshold:
            filtered_points.append(point)

    return np.array(filtered_points)

def visualize_geometries(geometries):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point Cloud Visualization', width=800, height=600)
    for geom in geometries:
        vis.add_geometry(geom)
    vis.run()
    vis.destroy_window()

def save_point_cloud(points, file_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(file_path, pcd)

def main():
    dense_points_path = 'danyepian1.ply'
    mesh_path = '/danyepian_mesh_final_filtered1.ply'
    threshold = 0.05

    dense_pcd = read_ply(dense_points_path)
    mesh = read_mesh(mesh_path)

    edge_points = get_edge_points_from_mesh(mesh)
    print(f"Number of edge points: {len(edge_points)}")

    dense_points = np.asarray(dense_pcd.points)

    if len(edge_points) > 0:
        edge_pcd = o3d.geometry.PointCloud()
        edge_pcd.points = o3d.utility.Vector3dVector(edge_points)
        edge_pcd.paint_uniform_color([1, 0, 0])
        mesh_pcd = o3d.geometry.PointCloud()
        mesh_pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        mesh_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        visualize_geometries([mesh, mesh_pcd, edge_pcd])
    else:
        mesh_pcd = o3d.geometry.PointCloud()
        mesh_pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        mesh_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        visualize_geometries([mesh, mesh_pcd])

    filtered_points = filter_dense_points(dense_points, edge_points, threshold)
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    visualize_geometries([filtered_pcd])

    filtered_edge_points_path = 'E:/Dataset/plant_model2/scan20_dianyunshuju/filtered_edge_points1_005.ply'
    save_point_cloud(filtered_points, filtered_edge_points_path)
    print(f"Filtered edge points saved to: {filtered_edge_points_path}")

    print(f"Original dense points: {len(dense_points)}")
    print(f"Filtered points: {len(filtered_points)}")

if __name__ == "__main__":
    main()