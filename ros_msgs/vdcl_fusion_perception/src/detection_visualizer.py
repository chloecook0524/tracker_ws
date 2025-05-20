#!/usr/bin/env python

import sys
import rospy
import torch
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from pyquaternion import Quaternion
from collections import deque
import matplotlib.pyplot as plt

from sensor_msgs.msg import PointCloud2, Image, CameraInfo, CompressedImage
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, Float32MultiArray
from vdcl_fusion_perception.msg import DetectionResult
from geometry_msgs.msg import Point


CAM_KEYS = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]

###############################################
# Visualization: 3D Marker + 6-Cam Mosaic
###############################################
class DetectionVisualizer(object):
    def __init__(self, cam_keys):
        self.detection_sub = rospy.Subscriber("/detection_results", DetectionResult, self.detection_callback)

        self.marker_pub = rospy.Publisher("/detection_markers", MarkerArray, queue_size=1)
        # 6캠 이미지를 모자이크 후 CompressedImage로 퍼블리시
        self.combined_pub = rospy.Publisher("/inference/combined_overlay/compressed", CompressedImage, queue_size=1)

        self.marker_id_counter = 0
        self.cam_keys = cam_keys

        # 모자이크 레이아웃 정의(2행×3열)
        self.rows = 2
        self.cols = 3
        assert len(cam_keys) == 6, "현재 예시는 6개 카메라 전제"

        self.label_to_dae = {
            0: "package://vdcl_fusion_perception/marker_dae/Car.dae",
            1: "package://vdcl_fusion_perception/marker_dae/Truck.dae",
            2: "package://vdcl_fusion_perception/marker_dae/Truck.dae",
            3: "package://vdcl_fusion_perception/marker_dae/Bus.dae",
            4: "package://vdcl_fusion_perception/marker_dae/Truck.dae",
            5: "package://vdcl_fusion_perception/marker_dae/Barrier.dae",
            6: "package://vdcl_fusion_perception/marker_dae/Motorcycle.dae",
            7: "package://vdcl_fusion_perception/marker_dae/Bicycle.dae",
            8: "package://vdcl_fusion_perception/marker_dae/Pedestrian.dae",
            9: "package://vdcl_fusion_perception/marker_dae/TrafficCone.dae",
        }
        # self.label_to_dae = {
        #     0: "package://vdcl_fusion_perception/marker_dae/Car.dae",
        #     1: "package://vdcl_fusion_perception/marker_dae/Car.dae",
        #     2: "package://vdcl_fusion_perception/marker_dae/Car.dae",
        #     3: "package://vdcl_fusion_perception/marker_dae/Car.dae",
        #     4: "package://vdcl_fusion_perception/marker_dae/Car.dae",
        #     5: "package://vdcl_fusion_perception/marker_dae/Car.dae",
        #     6: "package://vdcl_fusion_perception/marker_dae/Car.dae",
        #     7: "package://vdcl_fusion_perception/marker_dae/Car.dae",
        #     8: "package://vdcl_fusion_perception/marker_dae/Car.dae",
        #     9: "package://vdcl_fusion_perception/marker_dae/Car.dae",
        # }

        self.marker_array = MarkerArray()

    def _get_marker_id(self):
        self.marker_id_counter += 1
        self.marker_id_counter %= 1000000
        return self.marker_id_counter

    def mk_point(self, x, y, z):
        p = Point()
        p.x = x
        p.y = y
        p.z = z
        return p

    def create_3d_markers(self, bboxes_3d, score, label, stamp, frame_id, color=None):
        x, y, z, l, w, h, yaw, vx, vy = bboxes_3d  

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.id = self._get_marker_id()
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = self.label_to_dae[label]
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        quaternion = Quaternion(axis=[0, 0, 1], angle=yaw)
        marker.pose.orientation.w = quaternion[0]
        marker.pose.orientation.x = quaternion[1]
        marker.pose.orientation.y = quaternion[2]
        marker.pose.orientation.z = quaternion[3]
        marker.scale.x = l
        marker.scale.y = w
        marker.scale.z = h
        marker.color.a = min(score*5,1)
        if color is None:
            marker.color.r = 1
            marker.color.g = 1
            marker.color.b = 1
        else:
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
        self.marker_array.markers.append(marker)

        arrow = Marker()
        arrow.header.frame_id = frame_id
        arrow.header.stamp = stamp
        arrow.id = self._get_marker_id()
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.scale.x = 0.2
        arrow.scale.y = 0.5
        arrow.scale.z = 0.3
        arrow.color.a = 1
        arrow.color.r = 1
        arrow.color.g = 1
        arrow.color.b = 1
        arrow.points.append(self.mk_point(x, y, z+h/2))
        arrow.points.append(self.mk_point(x+vx, y+vy, z))
        self.marker_array.markers.append(arrow)

    # def publish_mosaic_compressed(self, imgs_tensor, bboxes_3d, lidar2img_mats, img_aug_mats, stamp):
    #     """
    #     6장 이미지를 (2행,3열) 모자이크 → 바운딩박스 그리기 → JPEG 압축 → CompressedImage로 퍼블리시.
    #     imgs_tensor shape=[1, 6, 3, H, W]
    #     """
    #     cam_imgs = imgs_tensor[0]  # (6,3,H,W)
    #     corners_all = bboxes_3d.corners.cpu().numpy()

    #     # 우선 개별 이미지를 numpy로 변환(각각 (H,W,3) in RGB)
    #     # 임의 해상도(H,W) 동일하다고 가정
    #     N, C, H, W = cam_imgs.shape

    #     # 모자이크 전체 높이=2*H, 너비=3*W
    #     mosaic_h = self.rows * H
    #     mosaic_w = self.cols * W
    #     # uint8 3채널 (RGB)
    #     mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

    #     # 카메라별로 2D 박스 투영 후 모자이크에 copy
    #     for idx, cam_key in enumerate(self.cam_keys):
    #         # sub-row, sub-col
    #         r = idx // self.cols
    #         c = idx % self.cols
    #         offset_y = r * H
    #         offset_x = c * W

    #         # 1) 이미지
    #         # shape=(3,H,W)->(H,W,3)
    #         img_np = cam_imgs[idx].cpu().numpy().transpose(1,2,0).astype(np.uint8)
    #         # 모자이크에 배치
    #         mosaic[offset_y:offset_y+H, offset_x:offset_x+W, :] = img_np

    #     # 2) 모자이크 위에 2D bounding box 그리기
    #     #    카메라별로 corner를 투영해, 모자이크에서의 위치로 변환.
    #     for i, cam_key in enumerate(self.cam_keys):
    #         r = i // self.cols
    #         c = i % self.cols
    #         offset_y = r * H
    #         offset_x = c * W

    #         # 해당 카메라의 투영행렬
    #         lidar2img = lidar2img_mats[i]
    #         aug_mat = img_aug_mats[i]

    #         for box_idx, corners_3d in enumerate(corners_all):
    #             corner_pts = np.concatenate([corners_3d, np.ones((8,1))], axis=1)
    #             proj = corner_pts @ lidar2img.T
    #             if (proj[:,2] <= 0).any():
    #                 continue
    #             proj[:,:2] /= proj[:,[2]]
    #             proj = proj @ aug_mat.T

    #             # 투영 후 실제 픽셀 좌표
    #             px = proj[:,0]
    #             py = proj[:,1]
    #             # 정수화
    #             px = px.astype(np.int32)
    #             py = py.astype(np.int32)

    #             # 모자이크 내부에만 그릴 것
    #             idx_pairs = [(0,1),(1,3),(3,2),(2,0),
    #                          (4,5),(5,7),(7,6),(6,4),
    #                          (0,4),(1,5),(2,6),(3,7)]
    #             for (p1, p2) in idx_pairs:
    #                 x1,y1 = px[p1], py[p1]
    #                 x2,y2 = px[p2], py[p2]
    #                 # 서브이미지에 대한 offset 적용
    #                 X1 = offset_x + x1
    #                 Y1 = offset_y + y1
    #                 X2 = offset_x + x2
    #                 Y2 = offset_y + y2
    #                 # 화면 범위 검사. (옵션적으로 skip 가능)
    #                 if not(0<=X1<mosaic_w and 0<=X2<mosaic_w and 0<=Y1<mosaic_h and 0<=Y2<mosaic_h):
    #                     continue
    #                 cv2.line(mosaic, (X1,Y1), (X2,Y2), (0,255,0), 1)

    #     # 3) 해상도 축소(옵션). 예: (mosaic_h//2, mosaic_w//2)
    #     #   필요하다면, 아래처럼 리사이즈해서 더 작게 만든 뒤 퍼블리시
    #     # mosaic = cv2.resize(mosaic, (mosaic_w//2, mosaic_h//2), interpolation=cv2.INTER_AREA)

    #     # 4) CompressedImage 메시지로 변환
    #     #   (RGB → JPEG)
    #     ret, buffer_jpg = cv2.imencode('.jpg', mosaic)  # np.ndarray
    #     if not ret:
    #         rospy.logwarn("Failed to encode mosaic image.")
    #         return
    #     compressed_msg = CompressedImage()
    #     compressed_msg.header.stamp = stamp
    #     compressed_msg.format = "jpeg"
    #     compressed_msg.data = np.array(buffer_jpg).tobytes()

    #     # 5) Publish
    #     self.combined_pub.publish(compressed_msg)

    def delete_all_markers(self):
        for i, marker in reversed(list(enumerate(self.marker_array.markers))):
            if marker.action == Marker.DELETE:
                del self.marker_array.markers[i]
            else:
                marker.action = Marker.DELETE

        self.marker_pub.publish(self.marker_array)

    def detection_callback(self, msg):
        self.delete_all_markers()

        stamp = msg.header.stamp
        frame_id = msg.header.frame_id

        results_3d = np.array(msg.result.data, dtype=np.float32).reshape(-1, 11)
        bboxes_3d = results_3d[:, :9]
        scores = results_3d[:, 9]
        labels = results_3d[:, 10].astype(np.int32)

        for bbox_3d, score, label in zip(bboxes_3d, scores, labels):
            # bboxes_3d shape=[x,y,z,l,w,h,yaw,vx,vy]
            marker = self.create_3d_markers(
                bboxes_3d=bbox_3d,
                score=score,
                label=label,
                stamp=stamp,
                frame_id=frame_id
            )
            # self.marker_array.markers.append(marker)

        marker = self.create_3d_markers(
            bboxes_3d=[0,0,-1.7,4,2,2,np.pi/2,0,0],
            score=1,
            label=0,
            stamp=stamp,
            frame_id=frame_id,
            color=[0,1,0]
        )

        # marker = self.create_3d_markers(
        #     bboxes_3d=[0,0,-1.7,4,2,2,0,0,0],
        #     score=1,
        #     label=0,
        #     stamp=stamp,
        #     frame_id=frame_id,
        #     color=[0,1,0]
        # )

        self.marker_pub.publish(self.marker_array)
        print("delay:", rospy.Time.now().to_sec() - stamp.to_sec())





if __name__ == '__main__':
    rospy.init_node('detection_visualizer', anonymous=True)
    visualizer = DetectionVisualizer(cam_keys=CAM_KEYS)

    # Spin to keep the script running
    try:
        rospy.spin()
    except KeyboardInterrupt:
        visualizer.delete_all_markers()
        print("Shutting down ROS Detection Visualizer.")
    finally:
        visualizer.delete_all_markers()
        print("Deleted all markers.")