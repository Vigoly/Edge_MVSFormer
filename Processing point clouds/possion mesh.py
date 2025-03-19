import open3d as o3d
import numpy as np

def remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """Remove statistical outliers from a point cloud."""
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_cloud = pcd.select_by_index(ind)
    return inlier_cloud

def remove_radius_outliers(pcd, nb_points=16, radius=0.05):
    """Remove radius outliers from a point cloud."""
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    inlier_cloud = pcd.select_by_index(ind)
    return inlier_cloud

def estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=500)):
    """Estimate normals for a point cloud."""
    pcd.estimate_normals(search_param=search_param)
    pcd.orient_normals_consistent_tangent_plane(k=30)
    return pcd

def poisson_reconstruction(pcd, depth=9, trim_value=0.1):
    """Perform Poisson surface reconstruction on a point cloud."""
    pcd = estimate_normals(pcd)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    # 修剪密度过低的区域
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, trim_value)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh

def filter_mesh_by_distance(mesh, point_cloud, distance_threshold=0.01):
    """Filter mesh vertices by distance to point cloud."""
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    vertices = np.asarray(mesh.vertices)
    indices_to_keep = []

    for i, vertex in enumerate(vertices):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 1)
        if np.linalg.norm(vertices[i] - point_cloud.points[idx[0]]) < distance_threshold:
            indices_to_keep.append(i)

    mesh_filtered = mesh.select_by_index(indices_to_keep)
    return mesh_filtered

def mesh_to_point_cloud(mesh, num_points):
    """Convert a mesh back to a point cloud."""
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
    return pcd

# 读取点云
input_path = "/all_torch_danyepian1.ply"
gt_point_cloud = o3d.io.read_point_cloud(input_path)
# o3d.visualization.draw_geometries([gt_point_cloud])
# 移除统计离群点
cleaned_point_cloud = remove_statistical_outliers(gt_point_cloud)

# 移除半径离群点
cleaned_point_cloud = remove_radius_outliers(cleaned_point_cloud)

# 对点云进行降采样
downsampled_point_cloud = cleaned_point_cloud.voxel_down_sample(voxel_size=0.01)

# 使用泊松重建生成光滑的 mesh
mesh = poisson_reconstruction(downsampled_point_cloud)

# 保存原始的 mesh
mesh_output_path = "/all_torch_mesh_danyepian1.ply"
o3d.io.write_triangle_mesh(mesh_output_path, mesh)
# 可视化 mesh 网格
o3d.visualization.draw_geometries([mesh], window_name="Poisson Reconstructed Mesh", width=800, height=600)
# 根据距离阈值过滤 mesh
filtered_mesh = filter_mesh_by_distance(mesh, cleaned_point_cloud)

# 保存过滤后的 mesh
filtered_mesh_output_path = "/danyepian_mesh_final_filtered1.ply"
o3d.io.write_triangle_mesh(filtered_mesh_output_path, filtered_mesh)

# 将mesh转换回点云
filled_point_cloud = mesh_to_point_cloud(filtered_mesh, num_points=len(cleaned_point_cloud.points))
# o3d.visualization.draw_geometries([filled_point_cloud])
# 保存降采样后的点云
downsampled_point_cloud_path = "/downsampled_point_cloud1.ply"
o3d.io.write_point_cloud(downsampled_point_cloud_path, downsampled_point_cloud)

# 保存最终点云
final_output_path = "/danyepian_final_filtered1.ply"
o3d.io.write_point_cloud(final_output_path, filled_point_cloud)

# 可视化
# o3d.visualization.draw_geometries([cleaned_point_cloud, filtered_mesh])

# 确保点云数量
print(f"Initial point cloud size: {len(gt_point_cloud.points)}")
print(f"Cleaned point cloud size: {len(cleaned_point_cloud.points)}")
print(f"Downsampled point cloud size: {len(downsampled_point_cloud.points)}")