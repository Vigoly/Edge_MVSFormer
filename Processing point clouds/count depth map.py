import numpy as np
import imageio
import os
from scipy.interpolate import griddata
from collections import defaultdict

CAM_WID, CAM_HGT = 1600,1200#重投影到的深度图尺寸
EPS = 1.0e-16
output_folder = '/'  # 指定保存深度图的目标文件夹

# 检查输出文件夹是否存在，如果不存在则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def save_png(depth_map_filename, data):
    # 保存深度图为 PNG 格式
    imageio.imwrite(depth_map_filename, (data * 255 / np.max(data)).astype(np.uint8))

for view_idx in range(48):
    zi = []

    cam_filename = f'{view_idx:08d}_cam.txt'
    cam_filepath = os.path.join('\cams/', cam_filename)

    with open(cam_filepath, 'r') as cam_file:
        lines = cam_file.readlines()

        # 找到外部参数和内部参数矩阵的起始行索引
        extrinsic_start = lines.index('extrinsic\n') + 1
        intrinsic_start = lines.index('intrinsic\n') + 1

        # 提取包含外部参数矩阵的行
        extrinsic_lines = lines[extrinsic_start:extrinsic_start + 4]
        extrinsic_matrix = np.zeros((4, 4), dtype=float)  # 创建一个 4x4 的 NumPy 数组

        for i, line in enumerate(extrinsic_lines):
            values = list(map(float, line.strip().split()))
            extrinsic_matrix[i, :] = values

        extrinsic_matrix[3, :] = [0, 0, 0, 1]


        intrinsic_lines = lines[intrinsic_start:intrinsic_start + 3]

        intrinsic_matrix = np.zeros((3, 3), dtype=float)  # 创建一个 3x3 的 NumPy 数组

        for i, line in enumerate(intrinsic_lines):
            values = list(map(float, line.strip().split()))
            intrinsic_matrix[i, :] = values


        txt_file_path = 'E:\Dataset\plant_model2\scan14_dianyunshuju/final_sampled_point - Cloud.txt'

        def point_cloud_to_camera_coords(pc, extrinsic_matrix):
            pc = np.asarray(pc)
            pc_selected = pc[:, :3]
            extended_pc = np.hstack((pc_selected, np.ones((pc.shape[0], 1))))
            points_camera_coords = np.dot(extrinsic_matrix, extended_pc.T).T
            return points_camera_coords[:, :3]

        def camera_coords_to_pixel_coords(points_camera_coords, intrinsic_matrix):
            points_image_coords = np.dot(intrinsic_matrix, points_camera_coords.T).T
            points_image_coords[:, 0] /= points_image_coords[:, 2]
            points_image_coords[:, 1] /= points_image_coords[:, 2]
            return points_image_coords[:, :2]



        pc = np.loadtxt(txt_file_path)


        points_camera_coords = point_cloud_to_camera_coords(pc, extrinsic_matrix)
        projected_points = camera_coords_to_pixel_coords(points_camera_coords, intrinsic_matrix)

        # 提取像素 x 和 y
        pixel_x = projected_points[:, 0]
        pixel_y = projected_points[:, 1]

        valid = np.bitwise_and(np.bitwise_and((pixel_x >= 0), (pixel_x < CAM_WID)),
                               np.bitwise_and((pixel_y >= 0), (pixel_y < CAM_HGT)))
        z = np.linalg.norm(points_camera_coords[valid, :3],axis=1)
        pixel_x, pixel_y = pixel_x[valid], pixel_y[valid],



        initial_depth = 2000  # 将初始深度设置为1000

        img_z = np.full((CAM_HGT, CAM_WID), initial_depth)
        img_z = img_z.astype(np.float32)

        for ui, vi, zi in zip(pixel_x, pixel_y, z):
            ui_int, vi_int = round(ui), round(vi)
            if 0 <= vi_int < img_z.shape[0] and 0 <= ui_int < img_z.shape[1]:
                if zi < img_z[vi_int, ui_int]:
                    img_z[vi_int, ui_int] = zi


        rolling_window_size = 3


        rolling_window = np.ones((rolling_window_size, rolling_window_size))


        for i in range(rolling_window_size):
            for j in range(rolling_window_size):
                img_z_shift = np.roll(img_z, (i - rolling_window_size // 2, j - rolling_window_size // 2), axis=(0, 1))
                img_z = np.minimum(img_z, img_z_shift)




        depth_map_filename = os.path.join(output_folder, f'depth_map_{view_idx:08d}.pfm')

        def save_pfm(depth_map_filename, data):


            with open(depth_map_filename, 'wb') as f:
                height, width = data.shape
                scale = 1.0  
                f.write(b'PF\n')
                f.write(f'{width} {height}\n'.encode())  # Convert to bytes
                f.write(f'{scale}\n'.encode())  # Convert to bytes
                data.tofile(f)


        # 保存 PNG 格式的深度图
        depth_map_png_filename = os.path.join(output_folder, f'png_depth_map_{view_idx:08d}.png')
        save_png(depth_map_png_filename, img_z)
        print(f"Saved PNG depth map for view {view_idx}.")

        # Check if depth is 1000 and replace it with 0
        img_z[img_z == 2000] = 0
        # Flip the depth map horizontally
        img_z = np.flip(img_z, axis=1)

        # Rotate the depth map by 90 degrees
        img_z = np.rot90(img_z, 2)

        print(f"Min depth: {np.min(img_z)}")
        print(f"Max depth: {np.max(img_z)}")

        # 保存深度图为.pfm格式
        save_pfm(depth_map_filename, img_z)
        print(f"Saved depth map for view {view_idx}.")