#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import time
import cv2
import numpy as np
import math

from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo, CompressedImage
import tf2_ros
from geometry_msgs.msg import TransformStamped, TwistStamped
# from cv_bridge import CvBridge


class NuScenesTimeReplayNode:
    def __init__(self):
        rospy.init_node("nuscene_time_replay_node", anonymous=True)

        # 1) NuScenes 로드 (예: mini, trainval 등)
        self.nusc = NuScenes(version='v1.0-trainval', dataroot='/home/sgsp/data/nuscenes', verbose=True)

        # 2) 씬(scene) 리스트 로드
        self.scenes = self.nusc.scene

        # 3) ROS Publisher 준비
        self.lidar_pub = rospy.Publisher("lidar_points", PointCloud2, queue_size=1)
        self.cam_keys = [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        self.cam_pubs = {}
        self.cam_info_pubs = {}
        for cam_key in self.cam_keys:
            self.cam_pubs[cam_key] = rospy.Publisher(f"{cam_key.lower()}", Image, queue_size=1)
            self.cam_info_pubs[cam_key] = rospy.Publisher(f"{cam_key.lower()}_info", CameraInfo, queue_size=1)  

        self.twist_pub = rospy.Publisher("ego_twist", TwistStamped, queue_size=1)
        self.last_ego_pose = None

        # self.cv_bridge = CvBridge()

    def spin(self):
        """
        전체 씬을 순회하면서, 각 씬의 모든 sample_data를 "정확한 시간차"로 재생.
        끝나면 처음 씬부터 무한반복.
        """
        while not rospy.is_shutdown():
            for scene in self.scenes:
                if rospy.is_shutdown():
                    break

                # 씬을 로드해서, 모든 sample_data(라이다+카메라) 정보를 "시간 순"으로 정렬
                replay_list = self.build_replay_list_for_scene(scene)

                # 1) 이 씬의 첫 번째 timestamp를 T0로 삼는다
                if len(replay_list) == 0:
                    continue
                t0 = replay_list[0]['timestamp']  # 가장 이른 시간 (마이크로초)

                # 2) 현재 시각(ROS 시간)을 기록, 재생 시작점
                start_ros_time = time.time()

                # 3) 순서대로 메시지 발행
                for i in range(len(replay_list)-1):
                    if rospy.is_shutdown():
                        break
                
                    self.publish_item(replay_list[i])

                    sleep_time = (replay_list[i+1]['timestamp'] - t0)/1e6 - (time.time() - start_ros_time)

                    time.sleep(max(sleep_time,0))

                rospy.loginfo(f"[Scene replay done] scene: {scene['name']}")
                time.sleep(1)  # 씬 간의 간격 (1초)

            # 모든 씬 끝나면 다시 처음부터
            rospy.loginfo("All scenes done. Restarting from first scene...")

        rospy.loginfo("Node shutdown complete.")

    # ----------------------------------------------------------------------
    # 씬(scene)에서 모든 sample_data(라이다+카메라) 가져오기
    # 시간(timestamp) 순으로 정렬한 뒤 리스트 반환
    # ----------------------------------------------------------------------
    def build_replay_list_for_scene(self, scene):
        replay_list = []
        sample_token = scene['first_sample_token']
        sample = self.nusc.get('sample', sample_token)

        sample = self.nusc.get('sample', sample_token)

        # 1) 라이다(LIDAR_TOP) sample_data
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_sd = self.nusc.get('sample_data', lidar_token)
        while True:
            replay_list.append({
                'sensor': 'lidar',
                'timestamp': lidar_sd['timestamp'],
                'sd_token': lidar_token
            })
            if lidar_sd['next'] == '':
                break
            lidar_token = lidar_sd['next']
            lidar_sd = self.nusc.get('sample_data', lidar_token)

        # 2) 카메라 6개
        for cam_key in self.cam_keys:
            cam_token = sample['data'][cam_key]
            cam_sd = self.nusc.get('sample_data', cam_token)
            while True:
                replay_list.append({
                    'sensor': cam_key,
                    'timestamp': cam_sd['timestamp'],
                    'sd_token': cam_token
                })
                if cam_sd['next'] == '':
                    break
                cam_token = cam_sd['next']
                cam_sd = self.nusc.get('sample_data', cam_token)

        # timestamp(마이크로초) 기준 오름차순 정렬
        replay_list.sort(key=lambda x: x['timestamp'])
        return replay_list

    # ----------------------------------------------------------------------
    # 개별 item(LiDAR or camera)을 Publish
    # item: { 'sensor': ..., 'timestamp': ..., 'sd_token': ... }
    # ----------------------------------------------------------------------
    def publish_item(self, item):
        sensor = item['sensor']
        sd_token = item['sd_token']

        if sensor == 'lidar':
            lidar_sd = self.nusc.get('sample_data', sd_token)
            lidar_cs = self.nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
            lidar_ego = self.nusc.get('ego_pose', lidar_sd['ego_pose_token'])
            pts = self.get_lidar_points(lidar_sd)
            pc2_msg = self.numpy_to_pointcloud2(pts)
            lidar_timestamp = rospy.Time.now()
            # lidar_timestamp = rospy.Time(lidar_sd['timestamp']/1e6)
            pc2_msg.header.stamp = lidar_timestamp
            pc2_msg.header.frame_id = "lidar"
            self.lidar_pub.publish(pc2_msg)
            # print("pub_delay:", rospy.Time.now().to_sec() - lidar_timestamp.to_sec())

            self.publish_transform_msg(
                lidar_cs['rotation'],
                lidar_cs['translation'],
                sensor_frame="lidar",
                parent_frame="ego_pose",
                timestamp=lidar_timestamp
            )

            # lidar_ego_timestamp = rospy.Time(lidar_ego['timestamp']/1e6)
            lidar_ego_timestamp = rospy.Time.now()

            self.publish_transform_msg(
                lidar_ego['rotation'],
                lidar_ego['translation'],
                sensor_frame="ego_pose",
                parent_frame="global",
                timestamp=lidar_ego_timestamp
            )

            self.publish_twist_msg(lidar_ego['translation'], lidar_ego['rotation'], lidar_ego_timestamp)

        elif sensor in self.cam_keys:
            cam_sd = self.nusc.get('sample_data', sd_token)
            cam_cs = self.nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
            cam_ego = self.nusc.get('ego_pose', cam_sd['ego_pose_token'])
            img_bgr = cv2.imread(f"{self.nusc.dataroot}/{cam_sd['filename']}")
            if img_bgr is not None:
                img_msg = self.numpy_to_image(img_bgr)
                # cam_timestamp = rospy.Time(cam_sd['timestamp']/1e6)
                cam_timestamp = rospy.Time.now()
                img_msg.header.stamp = cam_timestamp
                # img_msg.header.stamp = rospy.Time.now()
                img_msg.header.frame_id = sensor  # 예: "CAM_FRONT"
                self.cam_pubs[sensor].publish(img_msg)

                self.publish_transform_msg(
                    cam_cs['rotation'],
                    cam_cs['translation'],
                    sensor_frame=sensor.lower(),
                    parent_frame="ego_pose",
                    timestamp=cam_timestamp
                )
                cam_info_msg = self.get_camera_info(
                    cam_cs['camera_intrinsic'],
                    cam_sd['width'],
                    cam_sd['height'],
                    sensor,
                    timestamp=cam_timestamp
                )
                self.cam_info_pubs[sensor].publish(cam_info_msg)

                # cam_ego_timestamp = rospy.Time(cam_ego['timestamp']/1e6)
                cam_ego_timestamp = rospy.Time.now()
                self.publish_transform_msg(
                    cam_ego['rotation'],
                    cam_ego['translation'],
                    sensor_frame="ego_pose",
                    parent_frame="global",
                    timestamp=cam_ego_timestamp
                )

                # self.publish_twist_msg(cam_ego['translation'], cam_ego['rotation'], cam_ego_timestamp)
        else:
            pass  # 알 수 없는 센서 키면 무시

    def publish_twist_msg(self, ego_translation, ego_rotation, timestamp):
        if self.last_ego_pose is not None:
            last_timestamp, last_translation, last_rotation = self.last_ego_pose
            dt = (timestamp - last_timestamp).to_sec()
            if dt > 0.001:
                dx = ego_translation[0] - last_translation[0]
                dy = ego_translation[1] - last_translation[1]
                vx_global = dx / dt
                vy_global = dy / dt

                q = Quaternion(ego_rotation)
                yaw = q.yaw_pitch_roll[0]

                vx_local = math.cos(yaw) * vx_global + math.sin(yaw) * vy_global
                vy_local = -math.sin(yaw) * vx_global + math.cos(yaw) * vy_global

                q1 = Quaternion(last_rotation)
                q2 = Quaternion(ego_rotation)
                yaw1 = q1.yaw_pitch_roll[0]
                yaw2 = q2.yaw_pitch_roll[0]
                dyaw = yaw2 - yaw1
                dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
                yaw_rate = dyaw / dt

                twist_msg = TwistStamped()
                twist_msg.header.stamp = timestamp
                twist_msg.header.frame_id = "ego_pose"

                twist_msg.twist.linear.x = vx_local
                twist_msg.twist.linear.y = vy_local
                twist_msg.twist.angular.z = yaw_rate

                self.twist_pub.publish(twist_msg)

        self.last_ego_pose = (timestamp, ego_translation, ego_rotation)

    def publish_transform_msg(self, rotation, translation, sensor_frame, parent_frame, timestamp):
        """
        NuScenes에서 제공하는 quaternion rotation, 3D translation을
        4x4 변환 행렬(좌표계 변환)로 만들어주는 헬퍼 함수
        """
        br = tf2_ros.TransformBroadcaster()
        transform = TransformStamped()
        transform.header.stamp = timestamp
        transform.header.frame_id = parent_frame
        transform.child_frame_id = sensor_frame
        transform.transform.translation.x = translation[0]
        transform.transform.translation.y = translation[1]
        transform.transform.translation.z = translation[2]
        transform.transform.rotation.w = rotation[0]
        transform.transform.rotation.x = rotation[1]
        transform.transform.rotation.y = rotation[2]
        transform.transform.rotation.z = rotation[3]
        br.sendTransform(transform)

    def get_camera_info(self, intrinsics, width, height, frame_id, timestamp):
        """
        카메라 내부 파라미터를 ROS camera_info 메시지로 변환하여 발행하는 헬퍼 함수
        """
        camera_info = CameraInfo()
        camera_info.width = width
        camera_info.height = height
        camera_info.K = [intrinsics[i//3][i%3] for i in range(9)]
        camera_info.D = [0.0] * 5
        camera_info.distortion_model = "plumb_bob"
        camera_info.header.frame_id = frame_id
        camera_info.header.stamp = timestamp
        return camera_info

    # ----------------------------------------------------------------------
    # 라이다 포인트 불러오기 (단일 샘플 예시)
    # 스윕 붙이려면 기존 로직처럼 prev token 등 추적
    # ----------------------------------------------------------------------
    def get_lidar_points(self, lidar_sd):
        path = f"{self.nusc.dataroot}/{lidar_sd['filename']}"
        lidar_points = LidarPointCloud.from_file(path)
        pts_xyzi = lidar_points.points.T[:, :4]  # (x,y,z,intensity)
        return pts_xyzi

    # ----------------------------------------------------------------------
    # (N,5) -> sensor_msgs/PointCloud2
    # ----------------------------------------------------------------------
    def numpy_to_pointcloud2(self, points, frame_id='lidar'):
        pc2_msg = PointCloud2()
        pc2_msg.header.frame_id = frame_id
        pc2_msg.height = 1
        pc2_msg.width = points.shape[0]
        pc2_msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        pc2_msg.is_bigendian = False
        pc2_msg.point_step = 16
        pc2_msg.row_step = pc2_msg.point_step * points.shape[0]
        pc2_msg.is_dense = True

        pc2_msg.data = points.astype(np.float32).tobytes()
        return pc2_msg
    
    def numpy_to_image(self, img_np):
        """
        NumPy 배열을 ROS Image 메시지로 변환
        """
        img_msg = Image()
        img_msg.header.stamp = rospy.Time.now()
        img_np = img_np.astype(np.uint8)
        img_msg.height, img_msg.width = img_np.shape[:2]
        img_msg.encoding = "bgr8"
        img_msg.data = img_np.tobytes()

        # img_msg = CompressedImage()
        # img_msg.format = "jpeg"
        # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        # _, img_encoded = cv2.imencode('.jpg', img_np, encode_param)
        # img_msg.data = img_encoded.tobytes()
        return img_msg


if __name__ == "__main__":
    node = NuScenesTimeReplayNode()
    node.spin()
