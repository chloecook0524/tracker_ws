#!/usr/bin/env python3

import os
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from pypcd import pypcd

bag_file = '/home/sgsp/rosbag/20250410_refcar_LIDAR/refcar_LIDAR_2025-03-15-17-19-40.bag'  # rosbag 경로
topic_name = '/velodyne_FC/velodyne_points' 
output_dir = 'output_pcds'  

os.makedirs(output_dir, exist_ok=True)
bag = rosbag.Bag(bag_file)

for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[topic_name])):
    print(f"Processing frame {i}, timestamp: {t.to_sec()}")

    points = list(pc2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True))
    if not points:
        continue

    dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    np_points = np.array(points, dtype=dtype)

    pc = pypcd.PointCloud.from_array(np_points)

    pcd_path = os.path.join(output_dir, f"{i:06d}.pcd")
    pc.save_pcd(pcd_path, compression='ascii')  # or 'binary'
    print(f"Saved: {pcd_path}")
    break

bag.close()
