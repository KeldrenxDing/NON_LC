import os 
import sys
import time 
import copy 
from io import StringIO
import pypcd 
from pypcd import pypcd
import numpy as np
from numpy import linalg as LA
import open3d as o3d
from pypcdMyUtils import * 

####################################################
# 点云累积帧合并
####################################################

jet_table = np.load('jet_table.npy')
bone_table = np.load('bone_table.npy')

color_table = jet_table
color_table_len = color_table.shape[0]

data_dir = "/home/non2/SLAM/MY_TEST/LIO_NON/src/LIO_NON/save_PCD/" # 路径包含以/结束！！ 
scan_idx_range_to_stack = [0, len(scan_files)] # 存储的点云帧区间，自定义区间=[0, 8000]
node_skip = 1
num_points_in_a_scan = 150000 # 设置单帧点数量，MID360雷达使用150000
is_live_vis = False  
is_o3d_vis = True
intensity_color_max = 200
is_near_removal = True
thres_near_removal = 2 # 删除距离内结构特异点，单位：米
scan_dir = data_dir + "Scans"
scan_files = os.listdir(scan_dir) 
scan_files.sort()

# 导出的单帧存储文件夹目录，遍历设置 
poses = []
f = open(data_dir+"optimized_poses.txt", 'r')
while True:
    line = f.readline()
    if not line: break
    pose_SE3 = np.asarray([float(i) for i in line.split()])
    pose_SE3 = np.vstack( (np.reshape(pose_SE3, (3, 4)), np.asarray([0,0,0,1])) )
    poses.append(pose_SE3)
f.close()

assert (scan_idx_range_to_stack[1] > scan_idx_range_to_stack[0])
print("Merging scans from", scan_idx_range_to_stack[0], "to", scan_idx_range_to_stack[1])

if(is_live_vis):
    vis = o3d.visualization.Visualizer() 
    vis.create_window('Map', visible = True) 

nodes_count = 0
pcd_combined_for_vis = o3d.geometry.PointCloud()
pcd_combined_for_save = None

# miniPC内存预留
num_all_points_expected = int(num_points_in_a_scan * np.round((scan_idx_range_to_stack[1] - scan_idx_range_to_stack[0])/node_skip))

np_xyz_all = np.empty([num_all_points_expected, 3])
np_intensity_all = np.empty([num_all_points_expected, 1])
curr_count = 0

for node_idx in range(len(scan_files)):
    if(node_idx < scan_idx_range_to_stack[0] or node_idx >= scan_idx_range_to_stack[1]):
        continue

    nodes_count = nodes_count + 1
    if( nodes_count % node_skip is not 0): 
        if(node_idx is not scan_idx_range_to_stack[0]): # to ensure the vis init 
            continue

    print("read keyframe scan idx", node_idx)

    scan_pose = poses[node_idx]
    scan_path = os.path.join(scan_dir, scan_files[node_idx])
    scan_pcd = o3d.io.read_point_cloud(scan_path)
    scan_xyz_local = copy.deepcopy(np.asarray(scan_pcd.points))

    scan_pypcd_with_intensity = pypcd.PointCloud.from_path(scan_path)
    scan_intensity = scan_pypcd_with_intensity.pc_data['intensity']
    scan_intensity_colors_idx = np.round( (color_table_len-1) * np.minimum( 1, np.maximum(0, scan_intensity / intensity_color_max) ) )
    scan_intensity_colors = color_table[scan_intensity_colors_idx.astype(int)]

    scan_pcd_global = scan_pcd.transform(scan_pose) 
    scan_pcd_global.colors = o3d.utility.Vector3dVector(scan_intensity_colors)
    scan_xyz = np.asarray(scan_pcd_global.points)

    scan_intensity = np.expand_dims(scan_intensity, axis=1) 
    scan_ranges = LA.norm(scan_xyz_local, axis=1)

    if(is_near_removal):
        eff_idxes = np.where (scan_ranges > thres_near_removal)
        scan_xyz = scan_xyz[eff_idxes[0], :]
        scan_intensity = scan_intensity[eff_idxes[0], :]

        scan_pcd_global = scan_pcd_global.select_by_index(eff_idxes[0])

    if(is_o3d_vis):
        pcd_combined_for_vis += scan_pcd_global # open3d pointcloud class append is fast 

    if is_live_vis:
        if(node_idx is scan_idx_range_to_stack[0]): # to ensure the vis init 
            vis.add_geometry(pcd_combined_for_vis) 

        vis.update_geometry(pcd_combined_for_vis)
        vis.poll_events()
        vis.update_renderer()

    # 存储单帧
    np_xyz_all[curr_count:curr_count + scan_xyz.shape[0], :] = scan_xyz
    np_intensity_all[curr_count:curr_count + scan_xyz.shape[0], :] = scan_intensity

    curr_count = curr_count + scan_xyz.shape[0]
    print(curr_count)
#
if(is_o3d_vis):
    print(" 显示点云合并结果.")
    o3d.visualization.draw_geometries([pcd_combined_for_vis])
# 密度参数格式化，保存带密度元数据的点云
np_xyz_all = np_xyz_all[0:curr_count, :]
np_intensity_all = np_intensity_all[0:curr_count, :]
np_xyzi_all = np.hstack( (np_xyz_all, np_intensity_all) )
xyzi = make_xyzi_point_cloud(np_xyzi_all)
map_name = data_dir + "map_" + str(scan_idx_range_to_stack[0]) + "_to_" + str(scan_idx_range_to_stack[1]) + "_with_intensity.pcd"
xyzi.save_pcd(map_name, compression='binary_compressed')
print("intensity map is save (path:", map_name, ")")
