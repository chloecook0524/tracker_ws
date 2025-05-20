#!/usr/bin/env python3
import rospy
import numpy as np
import yaml
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import tf
import tf.transformations as tf_trans
from collections import deque
from typing import Tuple
import math


class LidarICPAligner:
    def __init__(self):
        self.master_topic = rospy.get_param('~master_topic', '/velodyne_FC/velodyne_points')
        self.slave_topics = rospy.get_param('~slave_topics', [
            '/velodyne_FL/velodyne_points', '/velodyne_FR/velodyne_points'])
            # '/velodyne_FR/velodyne_points'])
        self.frame_id = rospy.get_param('~frame_id', 'lidar')
        self.output_topic = rospy.get_param('~output_topic', '/aligned_cloud')
        self.time_threshold = rospy.get_param('~time_threshold', 0.05)
        self.buffer_size = rospy.get_param('~buffer_size', 30)
        self.save_path = rospy.get_param('~save_path', 'icp_results.yaml')

        # ✅ 1. 코드 내부에서 초기 extrinsic 설정
        master_extrinsic = [0.0, -0.5, -2.1, 1.45, 0.04, 1.73]
        self.initial_extrinsics = {
            '/velodyne_FL/velodyne_points': [0.5, -33.5,  88.0, 0.96, 0.52, 1.63],
            '/velodyne_FR/velodyne_points': [-0.5, 36.0, -89.0, 0.93, -0.52, 1.63],
        }

        # Buffers
        self.master_buffer = deque(maxlen=self.buffer_size)
        self.slave_buffers = {topic: deque(maxlen=self.buffer_size) for topic in self.slave_topics}

        # Transforms
        self.master_T = self.rpy_to_matrix(*master_extrinsic)
        self.transforms = {
            topic: np.linalg.inv(self.master_T) @ self.rpy_to_matrix(*self.initial_extrinsics.get(topic, [0, 0, 0, 0, 0, 0]))
            for topic in self.slave_topics
        }

        # Best results
        self.best_transforms = {topic: self.transforms[topic] for topic in self.slave_topics}
        self.best_fitness = {topic: -1.0 for topic in self.slave_topics}

        # ROS
        rospy.Subscriber(self.master_topic, PointCloud2, self.master_callback, queue_size=1, buff_size=2**24)
        for topic in self.slave_topics:
            rospy.Subscriber(topic, PointCloud2, self.slave_callback_gen(topic), queue_size=1, buff_size=2**24)

        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)
        self.tf_broadcaster = tf.TransformBroadcaster()

        rospy.on_shutdown(self.save_transforms)
        rospy.Timer(rospy.Duration(1.0), self.process)

    # def rpy_to_matrix(self, roll, pitch, yaw, x, y, z):
    #     T = tf_trans.euler_matrix(math.radians(roll), math.radians(pitch), math.radians(yaw))
    #     T[:3, 3] = [x, y, z]
    #     return T

    def rpy_to_matrix(self, roll_deg, pitch_deg, yaw_deg, x, y, z):
        # 라디안 변환
        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)

        # 각 축 회전 행렬 ( = intrinsic X→Y→Z 순서)
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ])
        Ry = np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])
        Rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Intrinsic 방식 (회전하는 축 기준) → R = Rz @ Ry @ Rx
        R = Rz @ Ry @ Rx

        # 변환 행렬 구성 (4x4)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T


    # def matrix_to_rpy_xyz(self, T):
    #     x, y, z = T[:3, 3]
    #     roll, pitch, yaw = tf_trans.euler_from_matrix(T)
    #     return [math.degrees(float(roll)), math.degrees(float(pitch)), math.degrees(float(yaw)), float(x), float(y), float(z)]

    def pc2_to_numpy(self, msg):
        # Define structured dtype
        dtype = np.dtype([
            ('timestampSec', np.uint32),   # 0-3
            ('timestampNsec', np.uint32),  # 4-7
            ('x', np.float32),             # 8-11
            ('y', np.float32),             # 12-15
            ('z', np.float32),             # 16-19
            ('intensity', np.uint8),       # 20
            ('ring', np.uint8),             # 21
            ('pad1', np.uint8),             # 22
            ('pad2', np.uint8),              # 23
        ])

        # Parse raw buffer into structured array
        points_raw = np.frombuffer(msg.data, dtype=dtype)

        # Extract x, y, z only
        points = np.vstack((points_raw['x'], points_raw['y'], points_raw['z'])).T

        return points.copy(), msg.header.stamp.to_sec()

    def numpy_to_o3d(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def publish_o3d_cloud(self, pcd):
        points = np.asarray(pcd.points)
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id
        msg = pc2.create_cloud_xyz32(header, points)
        self.pub.publish(msg)

    def master_callback(self, msg):
        pts, stamp = self.pc2_to_numpy(msg)
        self.master_buffer.append((stamp, pts))

    def slave_callback_gen(self, topic_name):
        def callback(msg):
            pts, stamp = self.pc2_to_numpy(msg)
            self.slave_buffers[topic_name].append((stamp, pts))
        return callback

    def process(self, event):
        if not self.master_buffer:
            return

        latest_master_stamp, latest_master_pts = self.master_buffer[-1]
        master_pcd = self.numpy_to_o3d(latest_master_pts)
        merged = o3d.geometry.PointCloud(master_pcd)
        slave_buffers = self.slave_buffers.copy()

        for topic, buffer in slave_buffers.items():
            if not buffer:
                continue

            stamps = [abs(s - latest_master_stamp) for s, _ in buffer]
            min_idx = int(np.argmin(stamps))
            if stamps[min_idx] > self.time_threshold:
                rospy.logwarn_throttle(5.0, f"[{topic}] No close match (Δt={stamps[min_idx]:.3f}s). Skipping.")
                continue

            slave_stamp, slave_pts = buffer[min_idx]
            slave_pcd = self.numpy_to_o3d(slave_pts)

            init = self.transforms[topic]
            transform, fitness, rmse = self.icp_align(slave_pcd, master_pcd, init)
            self.transforms[topic] = transform

            aligned = slave_pcd.transform(transform.copy())
            merged += aligned

            frame_name = topic.strip('/').split('/')[0]
            self.broadcast_transform(transform, frame_name)

            if fitness > self.best_fitness[topic]:
                self.best_fitness[topic] = fitness
                self.best_transforms[topic] = transform
                rospy.loginfo(f"[{frame_name}] 🏆 New best fitness: {fitness:.4f}, transform: {transform}")

            rospy.loginfo_throttle(2.0, f"[{frame_name}] ICP fitness={fitness:.3f}, RMSE={rmse:.3f}")

        # points = np.asarray(merged.points)
        # if points.shape[0] > 0:
        #     x_min, y_min, z_min = points.min(axis=0)
        #     x_max, y_max, z_max = points.max(axis=0)
        #     rospy.loginfo(f"[merged] x: {x_min:.2f} ~ {x_max:.2f}, "
        #                 f"y: {y_min:.2f} ~ {y_max:.2f}, "
        #                 f"z: {z_min:.2f} ~ {z_max:.2f}")
        # else:
        #     rospy.logwarn("merged.points is empty")
        merged = merged.voxel_down_sample(0.3)
        self.publish_o3d_cloud(merged)

    def icp_align(self, source_pcd, target_pcd, init_transform):
        # return init_transform, 0, 0
        source = source_pcd.voxel_down_sample(0.3)
        target = target_pcd.voxel_down_sample(0.3)
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

        reg = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance=1.0,
            init=init_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        return reg.transformation, reg.fitness, reg.inlier_rmse

    def broadcast_transform(self, transform, child_frame):
        trans = transform[:3, 3]
        quat = tf_trans.quaternion_from_matrix(transform)
        self.tf_broadcaster.sendTransform(
            trans, quat, rospy.Time.now(), child_frame, self.frame_id
        )

    def save_transforms(self):
        rospy.loginfo("Saving BEST transforms to YAML...")
        data = {}

        def matrix_to_dict(T):
            roll, pitch, yaw = tf_trans.euler_from_matrix(T)
            x, y, z = T[:3, 3]
            return {
                'x': round(float(x), 2),
                'y': round(float(y), 2),
                'z': round(float(z), 2),
                'roll': round(math.degrees(float(roll)), 2),
                'pitch': round(math.degrees(float(pitch)), 2),
                'yaw': round(math.degrees(float(yaw)), 2)
            }


        # ✅ 마스터 센서 extrinsic도 저장
        data['master'] = matrix_to_dict(self.master_T)

        # ✅ 각 슬레이브별 best transform 저장
        for topic, T in self.best_transforms.items():
            key = topic.strip('/').split('/')[0]
            global_T = self.master_T @ T
            print(f"{key}: {global_T}")
            data[key] = matrix_to_dict(global_T)

        with open(self.save_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        rospy.loginfo(f"Saved best ICP results to: {self.save_path}")



if __name__ == "__main__":
    rospy.init_node("lidar_icp_master_based_aligner")
    node = LidarICPAligner()
    rospy.spin()
