#!/usr/bin/env python3
import rospy
import numpy as np
import uuid
import json
from std_msgs.msg import Header, Float32
from lidar_processing_msgs.msg import LidarPerceptionOutput, LidarObject
from lidar_processing_msgs.msg import PfGMFATrack, PfGMFATrackArray
from scipy.optimize import linear_sum_assignment
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
import geometry_msgs.msg
import tf.transformations
import traceback
from typing import List, Dict
from shapely.geometry import Polygon
from tf.transformations import euler_from_quaternion
from lap import lapjv
import cv2
import math
from vdcl_fusion_perception.msg import DetectionObjects
from chassis_msgs.msg import Chassis
from pyquaternion import Quaternion
from geometry_msgs.msg import Point
from collections import deque


# # === Global Path to Baseversion Detection File ===
# BASE_DET_JSON = "/home/chloe/SOTA/MCTrack/data/base_version/nuscenes/centerpoint/val.json"
# GT_JSON_PATH = "/home/chloe/nuscenes_gt_valsplit.json"

# === BBox 클래스 (bbox.py 내용 통합) ===
class BBox:
    def __init__(self, frame_id, bbox, **kwargs):
        self.frame_id = frame_id
        self.is_fake = False
        self.is_interpolation = False
        self.category = bbox["category"]
        self.det_score = bbox["detection_score"]
        self.lwh = bbox["lwh"]
        self.global_xyz = bbox["global_xyz"]
        self.global_orientation = bbox["global_orientation"]
        self.global_yaw = bbox["global_yaw"]
        self.global_velocity = bbox["global_velocity"]
        self.global_acceleration = bbox["global_acceleration"]
        self.global_velocity_fusion = self.global_velocity
        self.global_acceleration_fusion = self.global_acceleration
        self.global_yaw_fusion = self.global_yaw
        self.lwh_fusion = self.lwh
        self.global_velocity_diff = [0, 0]
        self.global_velocity_curve = [0, 0]
        self.global_xyz_lwh_yaw = self.global_xyz + list(self.lwh) + [self.global_yaw]
        self.global_xyz_last = self.backward_prediction()
        self.global_xyz_lwh_yaw_last = self.global_xyz_last + list(self.lwh) + [self.global_yaw]
        self.global_xyz_lwh_yaw_predict = self.global_xyz_lwh_yaw
        self.global_xyz_lwh_yaw_fusion = self.global_xyz_lwh_yaw
        self.camera_type = bbox["bbox_image"].get("camera_type", None)
        self.x1y1x2y2 = bbox["bbox_image"].get("x1y1x2y2", [0.0, 0.0, 0.0, 0.0])
        self.x1y1x2y2_fusion = self.x1y1x2y2
        self.x1y1x2y2_predict = self.x1y1x2y2
        self.unmatch_length = 0

    def backward_prediction(self):
        last_xy = np.array(self.global_xyz[:2]) - np.array(self.global_velocity) * 0.5
        return last_xy.tolist() + [self.global_xyz[2]]
    
    def transform_bbox_tlbr2xywh(self, x1y1x2y2=None):
        if x1y1x2y2 is None:
            x1y1x2y2 = self.x1y1x2y2
        center_x = (x1y1x2y2[0] + x1y1x2y2[2]) / 2
        center_y = (x1y1x2y2[1] + x1y1x2y2[3]) / 2
        width = x1y1x2y2[2] - x1y1x2y2[0]
        height = x1y1x2y2[3] - x1y1x2y2[1]
        return np.array([center_x, center_y, width, height])

    def transform_bbox_xywh2tlbr(self, xywh):
        x1 = xywh[0] - (xywh[2] / 2)
        y1 = xywh[1] - (xywh[3] / 2)
        x2 = xywh[0] + (xywh[2] / 2)
        y2 = xywh[1] + (xywh[3] / 2)
        return np.array([x1, y1, x2, y2])

    def transform_3dbox2corners(self, global_xyz_lwh_yaw) -> np.ndarray:
        from pyquaternion import Quaternion  # ensure imported

        x, y, z, l, w, h, rot = global_xyz_lwh_yaw
        orientation = Quaternion(axis=[0, 0, 1], radians=rot)
        dx1, dx2, dy1, dy2, dz1, dz2 = (
            l / 2.0,
            l / 2.0,
            w / 2.0,
            w / 2.0,
            h / 2.0,
            h / 2.0,
        )
        x_corners = np.array([dx1, dx1, dx1, dx1, dx2, dx2, dx2, dx2]) * np.array(
            [1, 1, 1, 1, -1, -1, -1, -1]
        )
        y_corners = np.array([dy1, dy2, dy2, dy1, dy1, dy2, dy2, dy1]) * np.array(
            [1, -1, -1, 1, 1, -1, -1, 1]
        )
        z_corners = np.array([dz1, dz1, dz2, dz2, dz1, dz1, dz2, dz2]) * np.array(
            [1, 1, -1, -1, 1, 1, -1, -1]
        )
        corners = np.vstack((x_corners, y_corners, z_corners))
        corners = np.dot(orientation.rotation_matrix, corners)
        corners[0, :] += x
        corners[1, :] += y
        corners[2, :] += z
        return corners.T   

def box_to_polygon(x, y, size, yaw):
    corners = compute_bev_corners([x, y], size[:2], yaw)
    return Polygon(corners)


# def debug_track_vs_gt(tracks, gt_tracks, class_name):
#     # rospy.loginfo(f"▶ [DEBUG][{class_name.upper()}] GT 매칭 시작...")

#     class_id = [k for k, v in CLASS_NAME_MAP.items() if v == class_name]
#     if not class_id:
#         rospy.logwarn(f"[DEBUG] class_name={class_name} not found in CLASS_NAME_MAP")
#         return
#     class_id = class_id[0]

#     pred_tracks = [t for t in tracks if t["type"] == class_id]
#     gt_objs = [g for g in gt_tracks if g.get("tracking_name") == class_name]

#     for pt in pred_tracks:
#         px, py = pt["x"], pt["y"]
#         psize = pt["size"]
#         pyaw = pt["yaw"]
#         ppoly = box_to_polygon(px, py, psize, pyaw)

#         matched = False

#         for gt in gt_objs:
#             gx, gy = gt["translation"][:2]
#             gsize = gt["size"]
#             gyaw = quaternion_to_yaw(gt)
#             gpoly = box_to_polygon(gx, gy, gsize, gyaw)

#             dist = np.hypot(px - gx, py - gy)
#             iou = ppoly.intersection(gpoly).area / gpoly.union(ppoly).area if (ppoly.is_valid and gpoly.is_valid) else 0.0
#             yaw_error = np.abs(np.arctan2(np.sin(pyaw - gyaw), np.cos(pyaw - gyaw)))

#             rospy.loginfo(f"[{class_name.upper()}] Track ID={pt['id']} ↔ GT dist={dist:.2f}m, yaw_diff={yaw_error:.2f}rad, IoU={iou:.3f}")
#             rospy.loginfo(f"   ↪ Track size={psize}, yaw={pyaw:.2f}, GT size={gsize}, yaw={gyaw:.2f}")
#             if not ppoly.is_valid or not gpoly.is_valid:
#                 rospy.logwarn(f"[POLY-INVALID] Track ID={pt['id']} has invalid polygon.")
#             if dist < 10.0 and iou < 0.05:
#                 rospy.logwarn(f"[NO-MATCH] distance OK ({dist:.2f}m) but IoU too low → yaw misalign or size mismatch?")
#             if iou >= 0.5:
#                 matched = True
#                 break

#         # if not matched:
#         #     rospy.logwarn(f"[MISS] No matching GT for Track ID={pt['id']} (class={class_name})")

#     rospy.loginfo(f"▶ [DEBUG][{class_name.upper()}] 완료. 트랙 수={len(pred_tracks)}, GT 수={len(gt_objs)}")

# def compute_track_recall(tracks, gt_tracks, class_name, iou_thresh=0.5):
#     class_id = [k for k, v in CLASS_NAME_MAP.items() if v == class_name]
#     if not class_id:
#         return
#     class_id = class_id[0]

#     pred_tracks = [t for t in tracks if t["type"] == class_id]
#     gt_objs = [g for g in gt_tracks if g.get("tracking_name") == class_name]

#     matched_gt = set()
#     for pt in pred_tracks:
#         px, py = pt["x"], pt["y"]
#         psize = pt["size"]
#         pyaw = pt["yaw"]
#         ppoly = box_to_polygon(px, py, psize, pyaw)

#         for i, gt in enumerate(gt_objs):
#             gx, gy = gt["translation"][:2]
#             gsize = gt["size"]
#             gyaw = quaternion_to_yaw(gt)
#             gpoly = box_to_polygon(gx, gy, gsize, gyaw)
#             if not gpoly.is_valid or not ppoly.is_valid:
#                 continue
#             iou = ppoly.intersection(gpoly).area / gpoly.union(ppoly).area
#             if iou >= iou_thresh:
#                 matched_gt.add(i)
#                 break  # 한 GT와 매칭되면 종료

#     recall = len(matched_gt) / len(gt_objs) if gt_objs else 0.0
#     rospy.loginfo(f"[RECALL] {class_name} matched: {len(matched_gt)}/{len(gt_objs)} (Recall={recall:.3f})")

def cal_rotation_iou_inbev(pose1, pose2):
    import cv2
    box1 = np.array([pose1[0], pose1[1], pose1[3], pose1[4], pose1[6] * 180 / np.pi])
    box2 = np.array([pose2[0], pose2[1], pose2[3], pose2[4], pose2[6] * 180 / np.pi])
    base_xy = np.array(box1[:2])
    box1[:2] -= base_xy
    box2[:2] -= base_xy
    area1 = pose1[3] * pose1[4]
    area2 = pose2[3] * pose2[4]
    box1_inp = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
    box2_inp = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])

    int_pts = cv2.rotatedRectangleIntersection(tuple(box1_inp), tuple(box2_inp))[1]
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)
        union = cv2.contourArea(order_pts)
        iou = union / (area1 + area2 - union)
    else:
        iou = 0.0
        union = 0.0
    return iou, union

def compute_bev_corners(center, size, yaw):
    """
    3D 박스의 중심 (x, y), 크기 (w, l), yaw 회전값을 받아
    BEV 상의 4개 코너 좌표를 반환합니다. (회전 적용)
    """
    x, y = center
    w, l = size
    corners = np.array([
        [w/2, l/2],
        [w/2, -l/2],
        [-w/2, -l/2],
        [-w/2, l/2]
    ])

    rotation = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])

    rotated = corners @ rotation.T
    translated = rotated + np.array([x, y])
    return translated  # shape: (4, 2)

def polygon_iou(poly1, poly2):
    """
    두 사각형 폴리곤 (4x2 numpy array)의 IoU 계산
    (shapely 없이 수동 구현 가능하나, 간단히 shapely 사용 권장)
    """
    try:
        from shapely.geometry import Polygon
    except ImportError:
        raise ImportError("Please install shapely: pip install shapely")

    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0.0

def get_class_weights(class_id):
    weights = {
        1: (0.5, 1.5),  # car
        6: (1.0, 1.0),  # pedestrian (✓ keep)
        8: (1.0, 1.0),  # bicycle     ← 강화
        7: (1.0, 1.0),  # motorcycle  ← 강화
        3: (0.6, 0.4),  # bus         (✓ keep)
        4: (1.0, 1.0),  # trailer
        2: (1.0, 1.0),  # truck
    }
    return weights.get(class_id, (0.7, 0.3))


# === Utility Functions ===

def orientation_similarity(angle1_rad, angle2_rad):
    cosine_similarity = math.cos((angle1_rad - angle2_rad + np.pi) % (2 * np.pi) - np.pi)
    return (cosine_similarity + 1.0) / 2.0

def bbox_iou_2d(bbox1, bbox2):
    if not bbox1 or not bbox2:
        return 0.0
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0

def create_ego_marker(stamp):
    marker = Marker()
    marker.header.frame_id = "vehicle"
    marker.header.stamp = stamp
    marker.ns = "ego_vehicle"
    marker.id = 9999
    marker.type = Marker.MESH_RESOURCE
    marker.action = Marker.ADD
    marker.mesh_resource = "package://vdcl_fusion_perception/marker_dae/Car.dae"
    marker.mesh_use_embedded_materials = True

    marker.pose.position.x = 0.0
    marker.pose.position.y = 0.0
    marker.pose.position.z = 0.0

    quaternion = Quaternion(axis=[0, 0, 1], angle=0)
    marker.pose.orientation.w = quaternion[0]
    marker.pose.orientation.x = quaternion[1]
    marker.pose.orientation.y = quaternion[2]
    marker.pose.orientation.z = quaternion[3]

    marker.scale.x = 4.0
    marker.scale.y = 2.0
    marker.scale.z = 2.0

    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.lifetime = rospy.Duration(0.2)

    return marker

    
def create_single_track_marker(track, header, marker_id):
    m = Marker()
    m.header = header
    m.ns = "track_meshes"
    m.id = marker_id
    m.action = Marker.ADD
    m.type = Marker.MESH_RESOURCE
    m.mesh_use_embedded_materials = True
    m.pose.position.x = track["x"]
    m.pose.position.y = track["y"]
    
    # ✅ Z 위치 보정 (중심 기준)
    z_base = track["position"][2] if "position" in track and len(track["position"]) > 2 else 0.0
    z_center = z_base + track["size"][2] / 2.0
    m.pose.position.z = z_center

    q = tf.transformations.quaternion_from_euler(0, 0, track["yaw"])
    m.pose.orientation.x = q[0]
    m.pose.orientation.y = q[1]
    m.pose.orientation.z = q[2]
    m.pose.orientation.w = q[3]

    m.scale.x = track["size"][0]
    m.scale.y = track["size"][1]
    m.scale.z = track["size"][2]

    m.color.a = min(track["confidence"] * 5, 1.0)
    m.lifetime = rospy.Duration(0.0)
    m.color.r = 0.0
    m.color.g = 0.3
    m.color.b = 1.0

    class_mesh_paths = {
        1: "package://vdcl_fusion_perception/marker_dae/Car.dae",
        2: "package://vdcl_fusion_perception/marker_dae/Truck.dae",
        3: "package://vdcl_fusion_perception/marker_dae/Bus.dae",
        4: "package://vdcl_fusion_perception/marker_dae/Trailer.dae",
        5: "package://vdcl_fusion_perception/marker_dae/Truck.dae",
        6: "package://vdcl_fusion_perception/marker_dae/Pedestrian.dae",
        7: "package://vdcl_fusion_perception/marker_dae/Motorcycle.dae",
        8: "package://vdcl_fusion_perception/marker_dae/Bicycle.dae",
        9: "package://vdcl_fusion_perception/marker_dae/Barrier.dae",
        10: "package://vdcl_fusion_perception/marker_dae/TrafficCone.dae",
    }
    m.mesh_resource = class_mesh_paths.get(track["type"], "")
    return m

def create_text_marker(track, header, marker_id):
    t_m = Marker()
    t_m.header = header
    t_m.ns = "track_ids"
    t_m.id = marker_id
    t_m.type = Marker.TEXT_VIEW_FACING
    t_m.action = Marker.ADD
    t_m.pose.position.x = track["x"]
    t_m.pose.position.y = track["y"]
    t_m.pose.position.z = track["position"][2] + track["size"][2] + 2.5 if "position" in track and len(track["position"]) > 2 else track["size"][2] + 1.0
    t_m.scale.z = 0.8
    t_m.color.a = 1.0
    t_m.color.r = 1.0
    t_m.color.g = 1.0
    t_m.color.b = 1.0
    t_m.text = str(track["id"])
    return t_m

# def quaternion_to_yaw(obj):
#     """
#     GT object에서 rotation(quaternion) → yaw(rad) 로 변환
#     (NuScenes [w, x, y, z] 순서를 [x, y, z, w]로 변환 후 yaw 추출)
#     bicycle / motorcycle 클래스만 출력
#     """
#     rotation = obj.get("rotation", None)
#     name = obj.get("tracking_name", "")
    
#     if rotation and isinstance(rotation, list) and len(rotation) == 4:
#         try:
#             w, x, y, z = rotation
#             q = [x, y, z, w]
#             _, _, yaw = euler_from_quaternion(q)

#             # if name in ("bicycle", "motorcycle"):
#             #     rospy.loginfo(f"[quat→yaw] {name} → NuScenes q=[w={w:.3f}, x={x:.3f}, y={y:.3f}, z={z:.3f}] → yaw={yaw:.3f}")
            
#             return yaw
#         except Exception as e:
#             rospy.logwarn(f"[quat→yaw] {name} 변환 실패: {e}")
#             return 0.0
#     else:
#         if name in ("bicycle", "motorcycle"):
#             rospy.logwarn(f"[quat→yaw] {name} rotation 필드 없음 또는 포맷 오류")
#         return 0.0


# def create_gt_markers(gt_tracks, header):
#     markers = MarkerArray()
#     for i, obj in enumerate(gt_tracks):
#         m = Marker()
#         m.header = header
#         m.ns = "gt_boxes"
#         m.id = 2000 + i
#         m.type = Marker.CUBE
#         m.action = Marker.ADD
#         pos = obj.get('translation', [0,0,0])
#         size = obj.get('size', [1,1,1])
#         yaw = quaternion_to_yaw(obj) 

#         m.pose.position.x = pos[0]
#         m.pose.position.y = pos[1]
#         m.pose.position.z = 1.0  # 높이 중간

#         from tf.transformations import quaternion_from_euler
#         q = quaternion_from_euler(0, 0, yaw)
#         m.pose.orientation.x = q[0]
#         m.pose.orientation.y = q[1]
#         m.pose.orientation.z = q[2]
#         m.pose.orientation.w = q[3]

#         m.scale.x = size[0]
#         m.scale.y = size[1]
#         m.scale.z = size[2]

#         m.color.a = 0.4
#         m.color.r = 0.0
#         m.color.g = 0.5
#         m.color.b = 1.0

#         markers.markers.append(m)
#     return markers

def simple_matching_cost(track, det):
    """
    거리 + yaw 차이 기반 간단 매칭 cost.
    낮을수록 좋은 매칭.
    """
    dx = track.x[0] - det["position"][0]
    dy = track.x[1] - det["position"][1]
    dist = np.hypot(dx, dy)  # 2D 거리

    yaw_diff = abs(track.x[3] - det["yaw"])
    yaw_penalty = 1.0 - np.cos(yaw_diff)

    cost = dist + 0.3 * yaw_penalty  # weight는 조정 가능
    return cost

def cal_rotation_gdiou_inbev(box_trk, box_det, class_id, cal_flag=None):
    if cal_flag == "Predict":
        pose1 = box_trk.bboxes[-1].global_xyz_lwh_yaw_predict
        pose2 = box_det.global_xyz_lwh_yaw
    elif cal_flag == "BackPredict":
        pose1 = box_trk.bboxes[-1].global_xyz_lwh_yaw_fusion
        pose2 = box_det.global_xyz_lwh_yaw_last
    else:
        raise ValueError(f"Unexpected cal_flag value: {cal_flag}")

    corners1 = box_trk.bboxes[-1].transform_3dbox2corners(pose1)
    corners2 = box_det.transform_3dbox2corners(pose2)
    bev_idxes = [2, 3, 7, 6]
    bev_corners1 = corners1[bev_idxes, 0:2]
    bev_corners2 = corners2[bev_idxes, 0:2]
    pose1 = np.copy(pose1)
    pose2 = np.copy(pose2)
    iou, inter_area = cal_rotation_iou_inbev(pose1, pose2)
    union_points = np.concatenate([bev_corners1, bev_corners2], axis=0).astype(np.float64)
    union_points -= pose1[:2].reshape(1, 2)
    rect = cv2.minAreaRect(union_points.astype(np.float32))
    universe_area = rect[1][0] * rect[1][1]
    a_area = pose1[3] * pose1[4]
    b_area = pose2[3] * pose2[4]
    extra_area = universe_area - (a_area + b_area - inter_area)
    box_center_distance = (pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2
    union_distance = np.linalg.norm(np.array(rect[1])) ** 2
    box_trk_volume = pose1[3] * pose1[4] * pose1[5]
    box_det_volume = pose2[3] * pose2[4] * pose2[5]
    volume_ratio = (
        box_trk_volume / box_det_volume if box_trk_volume >= box_det_volume else box_det_volume / box_trk_volume
    )
    angle_ratio = orientation_similarity(pose1[6], pose2[6])
    w1, w2 = get_class_weights(class_id)

    ro_gdiou = (
        iou
        - w1 * extra_area / universe_area
        - w2 * box_center_distance / union_distance
    )
    
    # rospy.loginfo(f"[GDIoU] class={class_id}, IoU={iou:.3f}, "
    #             f"extra_area={extra_area:.3f}, center_dist={box_center_distance:.3f}, "
    #             f"volume_ratio={volume_ratio:.2f}, angle_sim={angle_ratio:.2f}, final={ro_gdiou:.3f}")

    return ro_gdiou

def sdiou_2d(bbox1, bbox2):
    """Scale-Dependent IoU"""
    if not bbox1 or not bbox2:
        return 0.0
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1])
    area2 = (bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1])
    union = area1 + area2 - inter
    if area1 == 0 or area2 == 0:
        rospy.logwarn(f"[SDIoU] Invalid area1={area1}, area2={area2}, bbox1={bbox1}, bbox2={bbox2}")
    iou = inter / union if union > 0 else 0.0

    cx1 = (bbox1[0] + bbox1[2]) / 2
    cy1 = (bbox1[1] + bbox1[3]) / 2
    cx2 = (bbox2[0] + bbox2[2]) / 2
    cy2 = (bbox2[1] + bbox2[3]) / 2
    center_dist = (cx1 - cx2)**2 + (cy1 - cy2)**2

    scale = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1]) + \
            (bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1])
    scale = max(scale, 1e-4)

    penalty = center_dist / (scale + 1e-6)
    return iou - 0.05 * penalty 


def bbox_iou_2d(bbox1, bbox2):
    if not bbox1 or not bbox2:
        return 0.0
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0

def ro_gdiou_2d(box1, box2, yaw1, yaw2):
    """
    MCTrack에서 사용하는 yaw-penalized IoU 유사 함수
    box1, box2: [w, l] (크기)
    yaw1, yaw2: 회전 각도 (radian)
    """
    w1, l1 = box1
    w2, l2 = box2

    if w1 <= 0 or l1 <= 0 or w2 <= 0 or l2 <= 0:
        return 0.0

    # 1. 일반 IoU 계산 (회전 무시, 축 정렬 bbox 가정)
    inter_w = min(w1, w2)
    inter_l = min(l1, l2)
    inter_area = inter_w * inter_l
    union_area = w1 * l1 + w2 * l2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0

    # 2. yaw 차이에 대한 cosine penalty
    yaw_diff = abs(yaw1 - yaw2)
    yaw_penalty = 1.0 - np.cos(yaw_diff)

    # 3. penalty 적용 GDIoU 유사 보정
    ro_gdiou = iou - 0.1 * yaw_penalty
    return max(0.0, ro_gdiou)    

# === Matching Threshold Helpers ===
def _get_class_distance_threshold(label):
    pedestrian_like = [6, 7, 8]
    return 1.5 if label in pedestrian_like else 3.0

def _get_reproj_iou_thresh(label):
    if label in [6, 7, 8]:  # pedestrian, motorcycle, bicycle
        return 0.2
    elif label in [1, 2, 3, 4]:  # car, truck, bus, trailer
        return 0.3
    return 0.25

# === Reprojection Matching Function ===
def image_plane_matching(tracks, detections):
    matches = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_dets = set(range(len(detections)))

    # cost_matrix 정의
    cost_matrix = np.zeros((len(tracks), len(detections)))  # 비용 행렬 초기화

    for ti, track in enumerate(tracks):
        best_iou, best_di = -1.0, -1
        for di, det in enumerate(detections):
            if det['type'] != track.label:
                continue
            bbox1 = getattr(track, 'reproj_bbox', None)
            bbox2 = det.get('reproj_bbox', None)
            if bbox1 is None or bbox2 is None:
                continue

            # 트래킹 객체와 디텍션 객체 간의 거리 계산
            dx = track.pose_state[0] - det["position"][0]  # track.x 대신 track.pose_state[0] 사용
            dy = track.pose_state[1] - det["position"][1]  # track.y 대신 track.pose_state[1] 사용
            dist = np.hypot(dx, dy)

            # 거리 기준이 특정 임계값을 초과하면 매칭을 고려하지 않음
            if dist > _get_class_distance_threshold(track.label):
                continue

            # IoU 계산 (또는 다른 계산 방식으로 비용 행렬을 업데이트)
            iou = bbox_iou_2d(bbox1, bbox2)
            threshold = _get_reproj_iou_thresh(track.label)

            # 비용 행렬에 IoU 값을 저장
            cost_matrix[ti, di] = 1 - iou  # IoU를 기반으로 비용 계산 (1 - IoU로 설정하여 IoU가 클수록 비용이 낮게 설정)

            # 최상의 매칭을 찾기 위해 조건을 비교
            if iou > best_iou and iou > threshold:
                best_iou = iou
                best_di = di

        if best_di >= 0:
            matches.append((ti, best_di))
            unmatched_tracks.discard(ti)
            unmatched_dets.discard(best_di)

    rospy.loginfo(f"[Image Matching] Attempted {len(tracks)} tracks vs {len(detections)} detections")
    rospy.loginfo(f"[Image Matching] Matched {len(matches)} tracks with detections.")
    return matches, list(unmatched_dets), list(unmatched_tracks)

def get_confirmed_bonus(label):
    """
    클래스별 CONFIRMED 트랙에 대한 cost 보너스 비율 반환
    작을수록 더 큰 보너스 (즉, cost 할인폭 큼)
    """
    if label in [1, 2, 3, 4]:  # 차량류
        return 0.7
    elif label in [6, 7, 8]:   # pedestrian, motorcycle, bicycle
        return 0.85
    else:
        return 0.8

def image_plane_matching_sdiou(tracks, detections):
    matches = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_dets = set(range(len(detections)))

    if len(tracks) == 0 or len(detections) == 0:
        return matches, list(unmatched_dets), list(unmatched_tracks)

    # cost_matrix 정의
    cost_matrix = np.zeros((len(tracks), len(detections)))  # 여기에 cost_matrix를 정의합니다.

    for ti, track in enumerate(tracks):
        best_iou, best_di = -1.0, -1
        for di, det in enumerate(detections):
            if det['type'] != track.label:
                continue
            bbox1 = getattr(track, 'reproj_bbox', None)
            bbox2 = det.get('reproj_bbox', None)
            if bbox1 is None or bbox2 is None:
                rospy.logwarn(f"[SDIoU] Missing bbox: track_id={track.id}, bbox1={bbox1}, bbox2={bbox2}")
                continue
            if bbox1 is not None and bbox2 is not None:
                iou_val = sdiou_2d(bbox1, bbox2)
                # rospy.loginfo(f"[RV-Match][SDIoU={iou_val:.3f}] bbox1={bbox1}, bbox2={bbox2}")

            dx = track.pose_state[0] - det["position"][0]  # track.x 대신 track.pose_state[0] 사용
            dy = track.pose_state[1] - det["position"][1]  # track.y 대신 track.pose_state[1] 사용
            dist = np.hypot(dx, dy)
            if dist > _get_class_distance_threshold(track.label):
                continue

            # IoU 계산
            sdiou = sdiou_2d(bbox1, bbox2)
            if sdiou <= 0:
                continue
            # rospy.loginfo(f"[SDIoU-Match] Track ID: {track.id}, Det idx: {di}, bbox1={bbox1}, bbox2={bbox2}, SDIoU: {sdiou:.3f}")
            threshold = _get_reproj_iou_thresh(track.label)
            
            # 비용 행렬 업데이트
            cost_matrix[ti, di] = 1.0 - sdiou

            if sdiou > best_iou and sdiou >= threshold:  # NOTE: threshold는 -0.3이므로 "크거나 같다" 체크
                best_iou = sdiou
                best_di = di

        if best_di >= 0:
            matches.append((ti, best_di))
            unmatched_tracks.discard(ti)
            unmatched_dets.discard(best_di)

    # 여기에 로그 추가
    # rospy.loginfo(f"IMAGEPLANESDIOU Cost Matrix: {cost_matrix}")  # 이제 cost_matrix가 정의되었으므로 로깅이 가능합니다.

    return matches, list(unmatched_dets), list(unmatched_tracks)


# === Hungarian IoU Matching Function with predicted boxes and distance-based cost ===
def hungarian_iou_matching(tracks, detections, use_hybrid_cost=False, dt=0.1, ego_vel=0.0, ego_yaw=0.0):
    if not tracks or not detections:
        rospy.logwarn("[Hungarian Matching] No tracks or detections available!")
        return [], list(range(len(detections))), list(range(len(tracks))), [], []

    VALID_CLASS_IDS = set(CLASS_CONFIG.keys())
    detections = [d for d in detections if d["type"] in VALID_CLASS_IDS]

    cost_thresholds = {
        1: 1.8,   # car
        2: 2.0,   # truck
        3: 2.2,   # bus
        4: 2.0,   # trailer
        5: 2.0,   # construction_vehicle
        6: 0.9,   # pedestrian
        7: 1.8,   # motorcycle
        8: 1.5,   # bicycle
        9: 1.0,   # barrier
        10: 0.8,  # traffic cone
    }
    default_threshold = 2.2

    cost_matrix = np.ones((len(tracks), len(detections))) * 1e6

    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            if det["type"] != track.label:
                continue

            pos_track = track.x[:2]
            pos_det = np.array(det["position"][:2])
            dist = np.linalg.norm(pos_track - pos_det)

            yaw_diff = abs(track.x[3] - det["yaw"])
            yaw_penalty = 1.0 - np.cos(yaw_diff)

            ro_iou = ro_gdiou_2d(track.size[:2], det["size"][:2], track.x[3], det["yaw"])
            iou_penalty = 1.0 - ro_iou

            if track.label in [6, 7, 8]:  # 작은 객체
                cost = 0.5 * dist + 0.3 * yaw_penalty + 0.2 * iou_penalty
            else:
                cost = 0.4 * dist + 0.3 * yaw_penalty + 0.3 * iou_penalty
            cost = 0.4 * dist + 0.3 * yaw_penalty + 0.3 * iou_penalty
            if track.label in [6, 7, 8]:
                with open("/tmp/mctrack_cost_debug.txt", "a") as f:
                    f.write(
                        f"[HYBRID_COST][CLS={track.label}] T#{i} ID={track.id} status={track.status_flag} "
                        f"traj_len={track.traj_length} missed={track.missed_count} → "
                        f"dist={dist:.2f}, yaw_diff={yaw_diff:.2f}, ro_gdiou={ro_iou:.3f}, cost={cost:.3f}\n"
                    )

            if hasattr(track, "status_flag") and track.status_flag == TrackState.CONFIRMED:
                cost *= get_confirmed_bonus(track.label)
            if cost > 10.0:
                continue

            cost_matrix[i, j] = cost

            # ✅ 로깅
            # with open("/tmp/mctrack_cost_debug.txt", "a") as f:
            #     f.write(
            #         f"[HYBRID_COST] T#{i} ID={track.id} vs D#{j} "
            #         f"→ dist={dist:.2f}, yaw_diff={yaw_diff:.2f}, ro_gdiou={ro_iou:.3f}, cost={cost:.3f}\n"
            #     )

        # === [7] 디버깅 로그 (선택) ===
        # with open("/tmp/mctrack_cost_debug.txt", "a") as f:
        #     f.write(f"[COST] T#{i} (ID={track.id}) vs D#{j} → dist={dist:.2f}, yaw_diff={yaw_diff:.2f}, cost={cost:.3f}\n")

            # with open("/tmp/mctrack_cost_debug.txt", "a") as f:
            #     f.write(f"[COST] T#{i} (ID={track.id}) vs D#{j} → IoU={iou_score:.3f}, dist={dist:.2f}, cost={cost_matrix[i,j]:.3f}\n")
    
    # === Hungarian Matching with Exception-Safe lapjv + Logging ===
    try:
        res = lapjv(cost_matrix, extend_cost=True, cost_limit=default_threshold)
        if not isinstance(res, tuple) or len(res) != 3:
            rospy.logwarn("[Hungarian] lapjv 결과 형식 오류")
            return [], list(range(len(detections))), list(range(len(tracks))), [], []
        total_cost, row_ind, col_ind = res
    except Exception as e:
        rospy.logerr(f"[Hungarian] lapjv failed: {e}")
        return [], list(range(len(detections))), list(range(len(tracks))), [], []

    matches = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_dets = set(range(len(detections)))

    for i, j in enumerate(row_ind):
        if j == -1 or i >= len(tracks) or j >= len(detections):
            continue
        cost = cost_matrix[i, j]
        label = tracks[i].label
        threshold = cost_thresholds.get(label, default_threshold)
        ro_iou = ro_gdiou_2d(tracks[i].size[:2], detections[j]["size"][:2], tracks[i].x[3], detections[j]["yaw"])

        if cost < threshold:
            matches.append((i, j))
            unmatched_tracks.discard(i)
            unmatched_dets.discard(j)
        else:
            if label in [6, 7, 8]:  # 소형 객체만 로깅
                log_line = (f"[HUNGARIAN FAIL] CLS={label} T#{i} ID={tracks[i].id} vs D#{j} "
                            f"→ dist={np.linalg.norm(tracks[i].x[:2] - np.array(detections[j]['position'][:2])):.2f}, "
                            f"yaw_diff={abs(tracks[i].x[3] - detections[j]['yaw']):.2f}, "
                            f"cost={cost:.3f}, threshold={threshold:.2f}, ro_gdiou={ro_iou:.3f}")
                with open("/tmp/mctrack_cost_debug.txt", "a") as f:
                    f.write(log_line + "\n")

    # for ti in unmatched_tracks:
    #     if ti < len(tracks):
    #         best_di = int(np.argmin(cost_matrix[ti]))
    #         best_cost = float(cost_matrix[ti][best_di])
            # rospy.loginfo(f"[UNMATCHED] T#{ti} vs best D#{best_di}: cost={best_cost:.2f}")
            # with open("/tmp/mctrack_cost_debug.txt", "a") as f:
            #     f.write(f"[UNMATCHED] T#{ti} vs best D#{best_di}: cost={best_cost:.2f}\n")

    matched_tracks = [tracks[i] for i, _ in matches]
    matched_detections = [detections[j] for _, j in matches]

    return matches, list(unmatched_dets), list(unmatched_tracks), matched_tracks, matched_detections

# === TrackState Enum ===
class TrackState:
    INITIALIZATION = 0
    CONFIRMED = 1
    OBSCURED = 2
    DEAD = 4

CLASS_NAME_MAP = {
    1: "car", 6: "pedestrian", 8: "bicycle", 7: "motorcycle", 3: "bus",
    4: "trailer", 2: "truck", 9: "barrier", 10: "cone", 5:"construction_vehicle",
}
# === 클래스별 칼만 필터 설정 (MCTRACK 완성 버전) ===
CLASS_CONFIG = {
    1: {  # car
        'confirm_threshold': 2,
        'max_unmatch': 3,
        'max_predict_len': 25,
        'confirmed_det_score': 0.7,
        'confirmed_match_score': 0.3,
        'is_filter_predict_box': -1,
        'expected_velocity': 10.0,
        'P': np.diag([1.0, 1.0, 10.0, 10.0]),
        'Q': np.diag([0.5, 0.5, 1.5, 1.5]),
        'R': np.diag([0.7, 0.7, 0.5, 0.5]),
        'P_size': np.eye(3),
        'Q_size': np.eye(3),
        'R_size': np.eye(2),
        'P_yaw': np.eye(2),
        'Q_yaw': np.eye(2),
        'R_yaw_scalar': 3.0,
    },
    6: {  # pedestrian
        'confirm_threshold': 2,                 # 기존 1
        'max_unmatch': 4,                       # 기존 1
        'max_predict_len': 25,                  # 기존 7 * 2.8
        'confirmed_det_score': 0.7,
        'confirmed_match_score': 0.3,
        'is_filter_predict_box': -1,
        'expected_velocity': 1.5,
        'P': np.diag([1.0, 1.0, 10.0, 10.0]),
        'Q': np.diag([0.5, 0.5, 1.5, 1.5]),
        'R': np.diag([0.7, 0.7, 1.0, 1.0]),
        'P_size': np.eye(3),
        'Q_size': np.eye(3),
        'R_size': np.eye(2),
        'P_yaw': np.eye(2),
        'Q_yaw': np.eye(2),
        'R_yaw_scalar': 1.5,
    },
    8:  {  # bicycle
        'confirm_threshold': 2,
        'max_unmatch': 3,
        'max_predict_len': 25,
        'confirmed_det_score': 0.7,
        'confirmed_match_score': 0.3,
        'is_filter_predict_box': -1,
        'expected_velocity': 7.0,
        'P': np.diag([1.0, 1.0, 1.0, 1.0]),
        'Q': np.diag([0.3, 0.3, 1.0, 1.0]),
        'R': np.diag([0.1, 0.1, 1.0, 1.0]),
        'P_size': np.eye(3),
        'Q_size': np.eye(3),
        'R_size': np.eye(2),
        'P_yaw': np.eye(2),
        'Q_yaw': np.eye(2),
        'R_yaw_scalar': 1.5,
    },
    7: {  # motorcycle
        'confirm_threshold': 2,
        'max_unmatch': 4,
        'max_predict_len': 30,
        'confirmed_det_score': 0.7,
        'confirmed_match_score': 0.3,
        'is_filter_predict_box': -1,
        'expected_velocity': 8.0,
        'P': np.diag([1.0, 1.0, 10.0, 10.0]),
        'Q': np.diag([0.5, 0.5, 4.0, 4.0]),
        'R': np.diag([0.3, 0.3, 1.0, 1.0]),
        'P_size': np.eye(3),
        'Q_size': np.eye(3),
        'R_size': np.eye(2),
        'P_yaw': np.eye(2),
        'Q_yaw': np.eye(2),
        'R_yaw_scalar': 2.5,
    },
    3:  {  # bus
        'confirm_threshold': 2,
        'max_unmatch': 3,
        'max_predict_len': 28,
        'confirmed_det_score': 0.7,
        'confirmed_match_score': 0.3,
        'is_filter_predict_box': -1,
        'expected_velocity': 6.0,
        'P': np.diag([100.0, 100.0, 100.0, 100.0]),
        'Q': np.diag([0.5, 0.5, 1.5, 1.5]),
        'R': np.diag([1.5, 1.5, 500.0, 500.0]),
        'P_size': np.eye(3),
        'Q_size': np.eye(3),
        'R_size': np.eye(2),
        'P_yaw': np.eye(2),
        'Q_yaw': np.eye(2),
        'R_yaw_scalar': 3.0,
    },
    4: {  # trailer
        'confirm_threshold': 2,
        'max_unmatch': 3,
        'max_predict_len': 20,
        'confirmed_det_score': 0.7,
        'confirmed_match_score': 0.3,
        'is_filter_predict_box': -1,
        'expected_velocity': 3.0,
        'P': np.diag([1.0, 1.0, 10.0, 10.0]),
        'Q': np.diag([0.3, 0.3, 0.1, 0.1]),
        'R': np.diag([2.0, 2.0, 2.5, 2.5]),
        'P_size': np.eye(3),
        'Q_size': np.eye(3),
        'R_size': np.eye(2),
        'P_yaw': np.eye(2),
        'Q_yaw': np.eye(2),
        'R_yaw_scalar': 3.0,
    },
    2: {  # truck
        'confirm_threshold': 2,
        'max_unmatch': 3,
        'max_predict_len': 30,
        'confirmed_det_score': 0.7,
        'confirmed_match_score': 0.3,
        'is_filter_predict_box': -1,
        'expected_velocity': 1.0,
        'P': np.diag([1.0, 1.0, 10.0, 10.0]),
        'Q': np.diag([0.1, 0.1, 2.0, 2.0]),
        'R': np.diag([1.5, 1.5, 4.0, 4.0]),
        'P_size': np.eye(3),
        'Q_size': np.eye(3),
        'R_size': np.eye(2),
        'P_yaw': np.eye(2),
        'Q_yaw': np.eye(2),
        'R_yaw_scalar': 3.0,
    },
    9: {  # barrier
        'confirm_threshold': 2,
        'max_unmatch': 3,
        'max_predict_len': 15,
        'confirmed_det_score': 0.5,
        'confirmed_match_score': 0.3,
        'is_filter_predict_box': -1,
        'expected_velocity': 0.0,  # 정적 객체로 가정
        'P': np.diag([1.0, 1.0, 10.0, 10.0]),
        'Q': np.diag([0.1, 0.1, 0.1, 0.1]),
        'R': np.diag([0.5, 0.5, 10.0, 10.0]),
        'P_size': np.eye(3),
        'Q_size': np.eye(3),
        'R_size': np.eye(2),
        'P_yaw': np.eye(2),
        'Q_yaw': np.eye(2),
        'R_yaw_scalar': 5.0,
    },
    10: {  # traffic cone
        'confirm_threshold': 2,
        'max_unmatch': 3,
        'max_predict_len': 15,
        'confirmed_det_score': 0.5,
        'confirmed_match_score': 0.3,
        'is_filter_predict_box': -1,
        'expected_velocity': 0.0,
        'P': np.diag([1.0, 1.0, 10.0, 10.0]),
        'Q': np.diag([0.1, 0.1, 0.1, 0.1]),
        'R': np.diag([0.5, 0.5, 10.0, 10.0]),
        'P_size': np.eye(3),
        'Q_size': np.eye(3),
        'R_size': np.eye(2),
        'P_yaw': np.eye(2),
        'Q_yaw': np.eye(2),
        'R_yaw_scalar': 5.0,
    },
    5: {  # construction_vehicle
        'confirm_threshold': 2,
        'max_unmatch': 3,
        'max_predict_len': 20,
        'confirmed_det_score': 0.5,
        'confirmed_match_score': 0.3,
        'is_filter_predict_box': -1,
        'expected_velocity': 3.0,
        'P': np.diag([1.0, 1.0, 10.0, 10.0]),
        'Q': np.diag([0.5, 0.5, 1.0, 1.0]),
        'R': np.diag([0.7, 0.7, 0.5, 0.5]),
        'P_size': np.eye(3),
        'Q_size': np.eye(3),
        'R_size': np.eye(2),
        'P_yaw': np.eye(2),
        'Q_yaw': np.eye(2),
        'R_yaw_scalar': 4.0,
    }
}

VALID_CLASSES = set(CLASS_CONFIG.keys())  # ex: {0, 1, 2, ..., 6}

# === KalmanTrackedObject (1/2) ===
class KalmanTrackedObject:
    def __init__(self, detection, obj_id=None):
        self.id = obj_id or (uuid.uuid4().int & 0xFFFF)
        self.label = detection['type']
        wlh = detection.get('size', [1.0, 1.0, 1.0])
        self.reproj_bbox = detection.get('reproj_bbox')
        self.traj_length = 1
        self.use_smoothing = False  
        self.status_flag = TrackState.INITIALIZATION 
        self.hits = 1
        self.yaw_drift_buffer = deque(maxlen=5)
        # self.state = TrackState.CONFIRMED if self.hits >= self.confirm_threshold else TrackState.TENTATIVE
        # self.status_flag = TrackState.INITIALIZATION
        px, py = detection['position'][:2]
        vx, vy = detection.get("velocity", [0.0, 0.0])
        self.pose_state = np.array([px, py, vx, vy])

        class_cfg = CLASS_CONFIG.get(self.label)
        if class_cfg is not None:
            self.pose_P = class_cfg["P"]
            self.pose_Q = class_cfg["Q"]
            self.pose_R = class_cfg["R"]
            self.size_P = class_cfg["P_size"]
            self.size_Q = class_cfg["Q_size"]
            self.size_R = class_cfg["R_size"]
            self.yaw_P = class_cfg["P_yaw"]
            self.yaw_Q = class_cfg["Q_yaw"]
            # self.yaw_R = class_cfg["R_yaw"]
            self.expected_velocity = class_cfg["expected_velocity"]
            self.confirm_threshold = class_cfg["confirm_threshold"]
            self.max_missed = class_cfg["max_unmatch"]
        else:
            self.pose_P = np.diag([1.0, 1.0, 10.0, 10.0])
            self.pose_Q = np.diag([0.5, 0.5, 1.5, 1.5])
            self.pose_R = np.diag([0.7, 0.7, 1.0, 1.0]) 
            self.size_P = np.diag([1.0, 1.0, 1.0])
            self.size_Q = np.diag([0.1, 0.1, 0.1])
            self.size_R = np.diag([0.1, 0.1, 0.1])
            self.yaw_P = np.diag([0.1, 0.1])
            self.yaw_Q = np.diag([0.1, 0.1])
            self.yaw_R = np.diag([0.2, 5.0])
            self.expected_velocity = 5.0
            self.confirm_threshold = 2
            self.max_missed = 3

        yaw = detection['yaw']
        self.yaw_state = np.array([yaw, 0.0])
        self.size_state = np.array(wlh[:3])
        self.age = 0.0
        self.missed_count = 0
        self.soft_deleted = False
        self.bboxes = []

    def predict(self, dt, ego_vel=0.0, ego_yaw_rate=0.0, ego_yaw=0.0):
        # === [1] Ego-motion Compensation ===
        if ego_vel > 0.01 or abs(ego_yaw_rate) > 0.01:
            dx = ego_vel * dt * np.cos(ego_yaw)
            dy = ego_vel * dt * np.sin(ego_yaw)
            dyaw = ego_yaw_rate * dt

            cos_yaw = np.cos(-dyaw)
            sin_yaw = np.sin(-dyaw)

            rel_x = self.pose_state[0]
            rel_y = self.pose_state[1]

            self.pose_state[0] = cos_yaw * rel_x - sin_yaw * rel_y - dx
            self.pose_state[1] = sin_yaw * rel_x + cos_yaw * rel_y - dy
            self.yaw_state[0] -= dyaw
            self.yaw_state[0] = np.arctan2(np.sin(self.yaw_state[0]), np.cos(self.yaw_state[0]))

        # === [2] Kalman Prediction (only if no detection) ===
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        self.pose_state = F @ self.pose_state
        self.pose_P = F @ self.pose_P @ F.T + self.pose_Q

        Fy = np.eye(2)
        Fy[0, 1] = dt
        self.yaw_state = Fy @ self.yaw_state
        self.yaw_P = Fy @ self.yaw_P @ Fy.T + self.yaw_Q
        self.yaw_state[0] = np.arctan2(np.sin(self.yaw_state[0]), np.cos(self.yaw_state[0]))

        self.size_P = self.size_P + self.size_Q

        self.age += dt
        # self.missed_count += 1

        confidence = max(0.0, min(1.0, self.tracking_score()))

        pred_dict = {
            "frame_id": -1,
            "category": self.label,
            "detection_score": confidence,
            "lwh": self.size.tolist(),
            "global_xyz": self.pose_state[:2].tolist() + [self.bboxes[-1].global_xyz[2] if self.bboxes else 0.0],
            "global_orientation": [0, 0, 0, 1],
            "global_yaw": self.yaw_state[0],
            "global_velocity": self.pose_state[2:].tolist(),
            "global_acceleration": [0.0, 0.0],
            "bbox_image": {"x1y1x2y2": [0, 0, 0, 0]}
        }

        pred_bbox = BBox(frame_id=pred_dict["frame_id"], bbox=pred_dict)
        pred_bbox.is_fake = True
        self.bboxes.append(pred_bbox)
        if len(self.bboxes) > 30:
            self.bboxes.pop(0)

        if self.status_flag == TrackState.CONFIRMED and self.missed_count > self.max_missed:
            # with open("/tmp/mctrack_cost_debug.txt", "a") as f:
            #     f.write(f"[STATE_CHANGE] ID={self.id} CONFIRMED → OBSCURED | missed={self.missed_count}, max={self.max_missed}\n")
            self.status_flag = TrackState.OBSCURED

        if self.status_flag == TrackState.OBSCURED and self.missed_count > (
            self.max_missed + CLASS_CONFIG[self.label]["max_predict_len"]
        ):
            # ✅ 삭제 보류 조건: traj_len > 10, confidence > 0.6
            if self.traj_length > 10 and getattr(self, 'confidence', 0.5) > 0.6:
                # with open("/tmp/mctrack_cost_debug.txt", "a") as f:
                #     f.write(f"[삭제 보류] ID={self.id}, missed={self.missed_count}, traj_len={self.traj_length}, conf={self.confidence:.2f}\n")
                return  # 아직 삭제 안 함

            # with open("/tmp/mctrack_cost_debug.txt", "a") as f:
            #     f.write(f"[STATE_CHANGE] ID={self.id} OBSCURED → DEAD | missed={self.missed_count}, max_predict_len={CLASS_CONFIG[self.label]['max_predict_len']}\n")
            self.status_flag = TrackState.DEAD


    def update(self, detection, dt, matched_score):
        pos = detection['position']
        vel = detection.get('velocity', [0.0, 0.0])

        # ✅ [Kalman Update] position 보정 (z = position[:2])
        z = np.array(pos[:2])
        H = np.eye(2, 4)  # position만 관측
        y = z - H @ self.pose_state
        R = self.pose_R[:2, :2] if hasattr(self, "pose_R") else np.diag([0.7, 0.7])
        S = H @ self.pose_P @ H.T + R
        K = self.pose_P @ H.T @ np.linalg.inv(S)
        self.pose_state = self.pose_state + K @ y
        self.pose_P = (np.eye(4) - K @ H) @ self.pose_P

        # ✅ 속도는 detection값 반영 (옵션: blending도 가능)
        alpha = 0.6
        self.pose_state[2:] = alpha * self.pose_state[2:] + (1 - alpha) * np.array(vel)

        # ✅ 불확실도 리셋 (선택 사항)
        self.pose_P[2:, 2:] = np.eye(2) * 1e-1

        # # ✅ Yaw 덮어쓰기 (간단화)
        # self.yaw_state[0] = detection['yaw']
        # self.yaw_state[1] = 0.0
        # self.yaw_P = np.eye(2) * 1e-2

        # === Yaw 제한 보정 with 튐 누적 보완 + 속도 기반 억제 ===
        def normalize_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        yaw_det = detection['yaw']
        yaw_diff = normalize_angle(yaw_det - self.yaw_state[0])
        self.yaw_drift_buffer.append(yaw_diff)
        if len(self.yaw_drift_buffer) > 5:
            self.yaw_drift_buffer.popleft()

        mean_drift = np.mean(self.yaw_drift_buffer)

        # ✅ 현재 속도에 따라 yaw 보정 계수 설정
        v = np.linalg.norm(self.pose_state[2:4])
        if v < 0.3:
            coeff = 0.0  # 정지: yaw 고정
        elif v < 2.0:
            coeff = 0.2  # 저속: 약한 보정
        else:
            coeff = 0.4  # 고속: 강한 보정

        if abs(mean_drift) < 0.5:
            self.yaw_state[0] += coeff * mean_drift
            self.yaw_state[0] = normalize_angle(self.yaw_state[0])
            self.yaw_state[1] = 0.0
            self.yaw_P = np.eye(2) * 1e-2
        elif len(self.yaw_drift_buffer) == 5 and all(abs(d) > 0.5 for d in self.yaw_drift_buffer):
            self.yaw_state[0] = yaw_det
            self.yaw_state[1] = 0.0
            self.yaw_P = np.eye(2) * 1e-2
            self.yaw_drift_buffer.clear()

        # # Yaw Kalman Update (안정화 버전)
        # z_yaw = detection["yaw"]
        # yaw_diff = np.arctan2(np.sin(z_yaw - self.yaw_state[0]), np.cos(z_yaw - self.yaw_state[0]))

        # if abs(yaw_diff) > 1.5:
        #     # 급격한 변화 → 덮어쓰기
        #     self.yaw_state[0] = z_yaw
        #     self.yaw_state[1] = 0.0
        #     self.yaw_P = np.eye(2) * 1e-2
        # else:
        #     # 정상 필터 업데이트
        #     H_yaw = np.array([[1, 0]])
        #     y = np.array([yaw_diff])
        #     R_yaw_scalar = 0.3
        #     Q_yaw = np.diag([0.005, 0.05])
        #     S = H_yaw @ self.yaw_P @ H_yaw.T + R_yaw_scalar
        #     K = self.yaw_P @ H_yaw.T / S
        #     self.yaw_state = self.yaw_state + (K.flatten() * y).flatten()
        #     self.yaw_state[0] = np.arctan2(np.sin(self.yaw_state[0]), np.cos(self.yaw_state[0]))
        #     self.yaw_state[1] = 0.0  # 안정화를 위해 yaw_rate는 무시
        #     self.yaw_P = (np.eye(2) - K @ H_yaw) @ self.yaw_P + Q_yaw

        # # ✅ Size 덮어쓰기
        # self.size_state[:2] = detection['size'][:2]
        # self.size_P = np.eye(3) * 1e-2

        # ✅ Size Kalman update (x, y, z에 대해 관측값 적용)
        z_size = np.array(detection['size'][:2]) 
        H_size = np.eye(2, 3) 
        y_size = z_size - H_size @ self.size_state
        S_size = H_size @ self.size_P @ H_size.T + self.size_R  # now (2x2)
        K_size = self.size_P @ H_size.T @ np.linalg.inv(S_size)
        self.size_state = self.size_state + K_size @ y_size
        self.size_P = (np.eye(3) - K_size @ H_size) @ self.size_P + self.size_Q

        # ✅ 기타 상태 업데이트
        self.soft_deleted = False
        self.missed_count = 0
        self.hits += 1
        self.traj_length += 1
        self.confidence = detection.get("confidence", 0.5)

        if self.traj_length > self.confirm_threshold or (
            self.confidence > CLASS_CONFIG[self.label]["confirmed_det_score"] and
            self.confidence > CLASS_CONFIG[self.label]["confirmed_match_score"]
        ):
            self.status_flag = TrackState.CONFIRMED

        self.reproj_bbox = detection.get('reproj_bbox')

        new_bbox = BBox(frame_id=detection.get("id", 0), bbox={
            "category": self.label,
            "detection_score": self.confidence,
            "lwh": detection["size"],
            "global_xyz": detection["position"][:3],
            "global_orientation": [0, 0, 0, 1],
            "global_yaw": detection["yaw"],
            "global_velocity": detection.get("velocity", [0.0, 0.0]),
            "global_acceleration": [0.0, 0.0],
            "bbox_image": {
                "x1y1x2y2": detection.get("reproj_bbox", [0.0, 0.0, 0.0, 0.0])
            }
        })
        self.bboxes.append(new_bbox)
        if len(self.bboxes) > 30:
            self.bboxes.pop(0)

        # update_log = (
        #     f"[UPDATE] ID={self.id}, hits={self.hits}, traj_len={self.traj_length}, "
        #     f"status={self.status_flag}, soft_deleted={self.soft_deleted}, missed={self.missed_count}"
        # )
        # with open("/tmp/mctrack_cost_debug.txt", "a") as f:
        #     f.write(update_log + "\n")

    def tracking_score(self):
        vel = np.hypot(self.pose_state[2], self.pose_state[3])
        expected_vel = self.expected_velocity
        vel_consistency = np.exp(-abs(vel - expected_vel) / (expected_vel + 1e-3))
        
        traj_penalty = np.exp(-0.2 * self.traj_length)  # trajectory 길이 penalty
        age_bonus = min(1.0, 0.1 * self.age)            # age 10초 이상이면 1.0으로 saturate

        score = (self.hits / (self.age + 1e-3)) * vel_consistency * (1.0 - traj_penalty) * age_bonus
        return max(0.1, min(1.0, score))

    @property
    def x(self):  # alias for existing usage
        return np.array([
            self.pose_state[0],
            self.pose_state[1],
            np.hypot(self.pose_state[2], self.pose_state[3]),
            self.yaw_state[0],
            self.yaw_state[1]
        ])

    @property
    def size(self):
        return self.size_state    

# === KalmanMultiObjectTracker (predict only) ===
class KalmanMultiObjectTracker:
    def __init__(self, use_hungarian=True, use_hybrid_cost=False):
        self.tracks = []
        self.use_hungarian = use_hungarian
        self.use_hybrid_cost = use_hybrid_cost
        self.use_rv_matching = False

    def predict(self, dt, ego_vel, ego_yaw_rate, ego_yaw):
        for t in self.tracks:
            t.predict(dt, ego_vel, ego_yaw_rate, ego_yaw)

    # === Modify Soft-deleted ReID with reproj_bbox filtering ===
    def _reid_soft_deleted_tracks(self,
                                unmatched_dets: List[int],
                                detections: List[Dict],
                                dt: float) -> List[int]:
        used = []
        for di in unmatched_dets:
            det = detections[di]
            label = det['type']

            if label not in CLASS_CONFIG:
                continue

            best_score = 0.0
            best_track = None
            for track in self.tracks:
                if track.label not in CLASS_CONFIG:
                    continue
                if track.traj_length > CLASS_CONFIG[track.label]["max_predict_len"]:
                    continue
                if not getattr(track, 'soft_deleted', False):
                    continue
                if track.label != label:
                    continue

                dx = track.pose_state[0] - det["position"][0]
                dy = track.pose_state[1] - det["position"][1]
                if np.hypot(dx, dy) > _get_class_distance_threshold(label):
                    continue

                score = ro_gdiou_2d(track.size[:2], det['size'][:2],
                                    track.yaw_state[0], det['yaw'])

                bbox1 = getattr(track, 'reproj_bbox', None)
                bbox2 = det.get('reproj_bbox', None)
                if bbox1 and bbox2:
                    iou = bbox_iou_2d(bbox1, bbox2)
                    if iou < 0.02:
                        # 예외로 허용하되 로그는 남김
                        with open("/tmp/mctrack_cost_debug.txt", "a") as f:
                            f.write(f"[REID WARN] ID={track.id} low IoU: {iou:.3f}, but continuing\n")

                if score <= 0.5:
                    # 완화, 다만 낮으면 경고
                    with open("/tmp/mctrack_cost_debug.txt", "a") as f:
                        f.write(f"[REID WARN] ID={track.id} low GDIoU: {score:.3f}\n")
                    continue

                if score > best_score:
                    best_score = score
                    best_track = track

            if best_track is not None:
                best_track.soft_deleted = False
                confidence = det.get("confidence", 0.5)
                best_track.update(det, dt, matched_score=confidence)
                best_track.hits += 1
                used.append(di)
                with open("/tmp/mctrack_cost_debug.txt", "a") as f:
                    f.write(f"[REID SUCCESS] ID={best_track.id} with det ID={di}, score={best_score:.3f}\n")

        return used


    def _fallback_match(self,
                        unmatched_trks: List[int],
                        unmatched_dets: List[int],
                        detections: List[Dict],
                        dt: float) -> List[int]:
        used = []
        for ti in unmatched_trks:
            track = self.tracks[ti]

            # ✅ 조건 1: CLASS 체크 + 너무 긴 트랙 무시
            if track.label not in CLASS_CONFIG:
                continue
            if track.traj_length > CLASS_CONFIG[track.label]["max_predict_len"]:
                continue

            # ✅ 조건 2: CONFIRMED 트랙은 fallback 금지 (Hungarian에서 책임져야 함)
            if track.status_flag == TrackState.CONFIRMED:
                continue

            # ✅ 조건 3: 트랙 길이가 너무 짧거나, missed가 없는 경우 무시
            if track.label in [6, 7, 8]:  # 소형 객체
                if track.traj_length < 2 or track.missed_count == 0:
                    continue
            else:
                if track.traj_length < 3 or track.missed_count == 0:
                    continue

            best_cost = float("inf")
            best_det = -1

            for di in unmatched_dets:
                if di in used:
                    continue
                det = detections[di]
                if det['type'] != track.label:
                    continue

                dx = track.pose_state[0] - det["position"][0]
                dy = track.pose_state[1] - det["position"][1]
                dist = np.hypot(dx, dy)
                if dist > _get_class_distance_threshold(track.label):
                    continue

                score = ro_gdiou_2d(track.size[:2], det['size'][:2],
                                    track.yaw_state[0], det['yaw'])

                cost = dist + (1.0 - score)

                # ✅ 조건 4: GDIoU score 기준 강화
                if score < 0.7:
                    continue

                if cost < 1.2 and cost < best_cost:
                    best_cost = cost
                    best_det = di

            if best_det >= 0:
                matched_det = detections[best_det]
                confidence = matched_det.get("confidence", 0.5)
                self.tracks[ti].update(matched_det, dt, matched_score=confidence)
                used.append(best_det)
            else:
                if track.label in [6, 7, 8]:  # 소형 객체만 추가 로깅
                    with open("/tmp/mctrack_cost_debug.txt", "a") as f:
                        f.write(
                            f"[FALLBACK FAIL][CLS={track.label}] T#{ti} ID={track.id} "
                            f"traj_len={track.traj_length}, missed={track.missed_count}, "
                            f"status={track.status_flag} → no match among {len(unmatched_dets)} dets\n"
                        )
        return used
    def update(self, detections, dt, ego_vel=0.0, ego_yaw_rate=0.0, ego_yaw=0.0):
        matched_tracks = []
        matched_dets = []
        # 0) predict
        for t in self.tracks:
            t.predict(dt, ego_vel, ego_yaw_rate, ego_yaw)

        # prepare unmatched lists
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(self.tracks)))

        # 1) Standard Hungarian matching
        if self.use_hungarian and self.tracks and detections:
            matches_h, unmatched_dets, unmatched_trks, matched_trks, matched_dets = \
                hungarian_iou_matching(self.tracks, detections, self.use_hybrid_cost,
                                    dt=dt, ego_vel=ego_vel, ego_yaw=ego_yaw)
            for tr, det in zip(matched_trks, matched_dets):
                tr.update(det, dt, matched_score=det.get("confidence", 0.5))

        # 🔽 여기에 추가 🔽
        for ti in unmatched_trks:
            track = self.tracks[ti]
            if track.traj_length <= 1 and track.missed_count > 0:
                track.soft_deleted = True
                with open("/tmp/mctrack_cost_debug.txt", "a") as f:
                    f.write(f"[DELETE EARLY] ID={track.id} traj_len={track.traj_length}, missed={track.missed_count} → soft_deleted\n")

        # 2) SDIoU (RV) matching — only if enabled
        if self.use_rv_matching and unmatched_trks and unmatched_dets:
            # pass only the leftovers
            tracks_for_s = [self.tracks[i] for i in unmatched_trks]
            dets_for_s   = [detections[i] for i in unmatched_dets]
            matches_s, new_unmatched_dets, new_unmatched_trks = image_plane_matching_sdiou(
                tracks_for_s, dets_for_s
            )
            # remap back to absolute indices
            rem_trks = [unmatched_trks[i] for i in new_unmatched_trks]
            rem_dets = [unmatched_dets[i]   for i in new_unmatched_dets]
            # update matched
            for rel_ti, rel_di in matches_s:
                abs_t = unmatched_trks[rel_ti]
                abs_d = unmatched_dets[rel_di]
                self.tracks[abs_t].update(detections[abs_d], dt)
            unmatched_trks = rem_trks
            unmatched_dets = rem_dets

        # 3) Fallback matching
        if unmatched_trks and unmatched_dets:
            used_f = self._fallback_match(unmatched_trks, unmatched_dets, detections, dt)
            # 여기서 used_f 에 포함된 det 인덱스는 이미 트랙이 업데이트됐으므로
            # unmatched_dets 에서 제거해야 합니다.
            unmatched_dets = [d for d in unmatched_dets if d not in used_f]
        
        # 4) Re-ID of soft-deleted tracks (conservative)
        if unmatched_dets:
            used_r = self._reid_soft_deleted_tracks(unmatched_dets, detections, dt)
            unmatched_dets = [d for d in unmatched_dets if d not in used_r]
        
        # 5) New track 생성
        num_new_tracks = 0
        for di in unmatched_dets:
            det = detections[di]
            new_track = KalmanTrackedObject(det)
            self.tracks.append(new_track)
            num_new_tracks += 1

        if num_new_tracks >= 10:
            with open("/tmp/mctrack_cost_debug.txt", "a") as f:
                f.write(f"[WARNING] {num_new_tracks} new tracks created in one frame!\n")
                
        # ✅ 6) missed_count 증가 먼저
        matched_track_ids = set(t.id for t in matched_tracks)
        for t in self.tracks:
            if t.id not in matched_track_ids:
                t.missed_count += 1
                # with open("/tmp/mctrack_cost_debug.txt", "a") as f:
                #     f.write(f"[MISSED_INC] ID={t.id}, missed_count={t.missed_count}\n")

        # ✅ 7) Soft-delete vs Hard-delete 판단 (missed_count 증가 반영됨)
        for t in self.tracks:
            if t.status_flag == TrackState.CONFIRMED:
                if t.missed_count > t.max_missed + 10:
                    t.soft_deleted = True
                    # with open("/tmp/mctrack_cost_debug.txt", "a") as f:
                    #     f.write(f"[SOFT_DELETE] ID={t.id}, status=CONFIRMED, missed={t.missed_count}, allowed={t.max_missed + 5}\n")
            else:
                if t.missed_count > t.max_missed:
                    t.soft_deleted = True
                    # with open("/tmp/mctrack_cost_debug.txt", "a") as f:
                    #     f.write(f"[SOFT_DELETE] ID={t.id}, status={t.status_flag}, missed={t.missed_count}, allowed={t.max_missed}\n")


        # # 8) 삭제 직전 로깅
        # for t in self.tracks:
        #     if t.soft_deleted and t.missed_count > t.max_missed + 10:
        #         with open("/tmp/mctrack_cost_debug.txt", "a") as f:
        #             f.write(f"[DELETE] ID={t.id} deleted | missed={t.missed_count}, soft_deleted={t.soft_deleted}, status={t.status_flag}\n")

        # 9) 실제 제거
        self.tracks = [
            t for t in self.tracks
            if not (t.soft_deleted and t.missed_count > t.max_missed + 3)
        ]

        # 🔻 여기에 추가하세요 🔻
        # with open("/tmp/mctrack_cost_debug.txt", "a") as f:
        #     for t in self.tracks:
        #         f.write(f"[STATE] ID={t.id}, cls={t.label}, hits={t.hits}, missed={t.missed_count}, traj_len={t.traj_length}, soft_deleted={t.soft_deleted}, status={t.status_flag}\n")
        # === 💬 CONFIRMED 상태의 차량 트랙 수 기록 ===
        confirmed_car_tracks = [t for t in self.tracks if t.label == 1 and t.status_flag == TrackState.CONFIRMED]
        # track_stat_log = f"[STATS] Confirmed car tracks = {len(confirmed_car_tracks)}"
        # with open("/tmp/mctrack_cost_debug.txt", "a") as f:
        #     f.write(track_stat_log + "\n")

        # 기존 출력 (필요시 유지)
        rospy.loginfo(f"[Tracker] Total Tracks: {len(self.tracks)}")

    def get_tracks(self):
        results = []

        # ✅ 평가 대상 클래스만 기록 (NuScenes 기준)
        VALID_LOG_LABELS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10
                            }

        for t in self.tracks:
            if CLASS_CONFIG[t.label].get("is_filter_predict_box", -1) == 1 and t.hits == 0:
                continue
            if t.status_flag != TrackState.CONFIRMED:
                continue
            if getattr(t, 'soft_deleted', False):
                continue
            if getattr(t, 'traj_length', 0) < 1:
                continue
            if t.label not in VALID_LOG_LABELS:
                continue

            # # ✅ 여기 추가: 특정 좌표에 있는 트랙만 로깅
            # FIXED_X, FIXED_Y = -12.9, 4.8  # 대략 찍은 위치 근처
            # if t.label == 1 and abs(t.x[0] - FIXED_X) < 0.5 and abs(t.x[1] - FIXED_Y) < 0.5:
            #     with open("/tmp/track_id_at_location.txt", "a") as f:
            #         f.write(f"[FIXED_TRACK] ID={t.id}, x={t.x[0]:.2f}, y={t.x[1]:.2f}, traj_len={t.traj_length}\n")

            # 기존 트랙 결과에 추가
            x, y, yaw = t.x[0], t.x[1], t.x[3]
            size = t.size
            score = t.confidence
            results.append({
                "id":         t.id,
                "x":          x,
                "y":          y,
                "yaw":        yaw,
                "size":       size,
                "confidence": score,
                "type":       t.label,
                "velocity":   t.pose_state[2:4].tolist(),
                "position":   [x, y, t.bboxes[-1].global_xyz[2] if t.bboxes else 0.0]
            })

        return results
    

class MCTrackTrackerNode:
    LABEL_STR_TO_ID = {
        "car": 1,
        "truck": 2,
        "bus": 3,
        "trailer": 4,
        "construction vehicle": 5,
        "pedestrian": 6,
        "motorcycle": 7,
        "bicycle": 8,
        "barrier": 9,
        "traffic cone": 10,
    }
    def __init__(self):
        # 1) 반드시 init_node 부터 호출
        rospy.init_node("mctrack_tracker_node", anonymous=True)
        self.tracking_timer = rospy.Timer(rospy.Duration(0.1), self.tracking_publish_callback)  # 10Hz
        # rospy.loginfo("[Tracker] 초기화 시작")

        # # 2) logger_ready 파라미터가 올라올 때까지 대기 (최대 15초)
        # rospy.loginfo("[Tracker] /logger_ready 기다리는 중…")
        # start = rospy.Time.now()
        # while not rospy.has_param("/logger_ready") and (rospy.Time.now() - start) < rospy.Duration(15.0):
        #     if rospy.is_shutdown():
        #         return
        #     rospy.sleep(0.1)
        # if rospy.has_param("/logger_ready"):
        #     rospy.loginfo("[Tracker] /logger_ready 감지, 시작합니다")
        # else:
        #     rospy.logwarn("[Tracker] /logger_ready 대기 타임아웃, 계속 진행")

        # # 3) GT JSON 로드
        # with open(GT_JSON_PATH, 'r') as f:
        #     raw = json.load(f)
        # raw_results = raw.get("results", {})
        # if isinstance(raw_results, dict):
        #     self.gt_data = raw_results
        # else:
        #     self.gt_data = {}
        #     for ann in raw_results:
        #         token = ann.get("sample_token") or ann.get("token")
        #         if token:
        #             self.gt_data.setdefault(token, []).append(ann)

        # self.total_frames = len(self.gt_data)
        self.frame_idx    = 0
        self.start_time   = rospy.Time.now()
        self.marker_array = MarkerArray()
        self.prev_track_ids = set()  # 트랙 ID 관리
        
        # # === Static TF (map → base_link) 퍼블리시 세팅 ===
        # self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        # self.publish_static_tf()

        # 4) Kalman 트래커 초기화
        use_rv_matching = False   
        use_hybrid = True   

        self.tracker = KalmanMultiObjectTracker(
            use_hungarian=True,
            use_hybrid_cost=use_hybrid
        )
        self.tracker.use_rv_matching = use_rv_matching
        self.tracker.use_confidence_filtering = True
        # 5) 퍼블리셔 생성 & 구독자 연결 대기
        self.tracking_pub = rospy.Publisher("/tracking/objects",
                                            PfGMFATrackArray,
                                            queue_size=100)
        self.vis_pub = rospy.Publisher("/tracking/markers", MarkerArray, queue_size=10)                                    
        # rospy.loginfo("[Tracker] /tracking/objects 구독자 기다리는 중…")
        # while self.tracking_pub.get_num_connections() == 0 and not rospy.is_shutdown():
        #     rospy.sleep(0.1)
        # rospy.loginfo("[Tracker] 구독자 연결 완료, 리플레이어 구독 시작")

        # 6) 리플레이어 콜백 구독
        self.detection_sub = rospy.Subscriber("/detection_objects",
                                              DetectionObjects,
                                              self.detection_callback,
                                              queue_size= 500,
                                              tcp_nodelay=True)
        # self.vel_sub     = rospy.Subscriber("/ego_vel_x", Float32, self.vel_callback, queue_size=1)
        # self.yawrate_sub = rospy.Subscriber("/ego_yaw_rate", Float32, self.yawrate_callback, queue_size=1)
        # self.yaw_sub     = rospy.Subscriber("/ego_yaw", Float32, self.yaw_callback, queue_size=1)
        self.chassis_sub = rospy.Subscriber("/chassis", Chassis, self.chassis_callback, queue_size=100)

        self.chassis_buffer = deque(maxlen=100) 
        # 7) ego state 초기화 & 이전 타임스탬프 변수
        self.ego_vel         = 0.0
        self.ego_yaw_rate    = 0.0
        self.ego_yaw         = 0.0
        self.last_time_stamp = None
        # self.last_token = None
        self.marker_timer = rospy.Timer(rospy.Duration(0.1), self.visualization_timer_callback)

        rospy.loginfo("MCTrackTrackerNode 초기화 완료.")

    def vel_callback(self, msg):
        self.ego_vel = msg.data

    def yawrate_callback(self, msg):
        self.ego_yaw_rate = msg.data

    def yaw_callback(self, msg):
        self.ego_yaw = msg.data

    def chassis_callback(self, msg):
        stamp_sec = msg.header.stamp.to_sec()  # ✅ 이 줄 추가
        self.chassis_buffer.append((stamp_sec, msg))
    
    def get_nearest_chassis(self, target_stamp):
        if not self.chassis_buffer:
            return None
        times = [abs(t - target_stamp) for (t, _) in self.chassis_buffer]
        nearest_idx = int(np.argmin(times))
        return self.chassis_buffer[nearest_idx][1]

    def visualization_timer_callback(self, event):
        # 10Hz 주기로 마커 퍼블리시만 담당
        if hasattr(self, "marker_array"):
            self.vis_pub.publish(self.marker_array)

    def tracking_publish_callback(self, event):
        if not hasattr(self, 'last_time_stamp'):
            return  # 아직 첫 detection이 안 들어왔으면 skip

        tracks = self.tracker.get_tracks()

        ta = PfGMFATrackArray()
        ta.header.stamp = rospy.Time.now()  # 또는 self.last_time_stamp
        ta.header.frame_id = "vehicle"

        for t in tracks:
            m = PfGMFATrack()
            m.pos_x = t["x"]
            m.pos_y = t["y"]
            m.yaw   = t["yaw"]
            dims    = list(t["size"])[:3]
            m.boundingbox = dims + [0.0] * 5
            m.confidence_ind = t["confidence"]
            m.id = int(t["id"])
            m.obj_class = t["type"]
            ta.tracks.append(m)

        self.tracking_pub.publish(ta)

    def delete_all_markers(self):
        for i, marker in reversed(list(enumerate(self.marker_array.markers))):
            if marker.action == Marker.DELETE:
                del self.marker_array.markers[i]
            else:
                marker.action = Marker.DELETE
        self.vis_pub.publish(self.marker_array)
        self.marker_array = MarkerArray()  # 비우기

    def publish_static_tf(self):
        static_tf = geometry_msgs.msg.TransformStamped()
        static_tf.header.stamp = rospy.Time.now()
        static_tf.header.frame_id = "map"
        static_tf.child_frame_id = "base_link"
        static_tf.transform.translation.x = 0.0
        static_tf.transform.translation.y = 0.0
        static_tf.transform.translation.z = 0.0
        q = tf.transformations.quaternion_from_euler(0, 0, 0)
        static_tf.transform.rotation.x = q[0]
        static_tf.transform.rotation.y = q[1]
        static_tf.transform.rotation.z = q[2]
        static_tf.transform.rotation.w = q[3]
        self.static_broadcaster.sendTransform(static_tf)

        # rospy.loginfo("🛰️ [Tracker] Static TF (map → base_link) published.")    

    def detection_callback(self, msg):
        try:
            timestamp_sec = msg.header.stamp.to_sec()
            nearest_chassis_msg = self.get_nearest_chassis(timestamp_sec)
            if nearest_chassis_msg:
                avg_speed_kph = (nearest_chassis_msg.whl_spd_fl + nearest_chassis_msg.whl_spd_fr +
                                nearest_chassis_msg.whl_spd_rl + nearest_chassis_msg.whl_spd_rr) / 4.0
                self.ego_vel = avg_speed_kph / 3.6
                self.ego_yaw_rate = nearest_chassis_msg.cr_yrs_yr * np.pi / 180.0
            # rospy.logdebug(f"[DEBUG] detection_callback 시작 (timestamp={timestamp_sec:.6f}, 객체 수={len(msg.objects)})")

            # 1) dt 계산
            if self.last_time_stamp is None:
                dt = 0.0
            else:
                dt = (msg.header.stamp - self.last_time_stamp).to_sec()
            self.last_time_stamp = msg.header.stamp

            self.frame_idx += 1
            rospy.loginfo(f"[Tracker] Frame {self.frame_idx} (dt={dt:.3f}s)")

            # 2) 동일 timestamp 반복 수신 방지
            if hasattr(self, "last_timestamp_sec") and abs(timestamp_sec - self.last_timestamp_sec) < 1e-6:
                rospy.logwarn(f"[WARN] 동일한 timestamp가 반복 수신됨: {timestamp_sec:.6f}")
            self.last_timestamp_sec = timestamp_sec

            # 3) Detection 변환 (VALID_CLASSES 필터링 포함)
            VALID_CLASSES = set(CLASS_CONFIG.keys())
            # class_min_confidence = {k: 0.0 for k in VALID_CLASSES}
            class_min_confidence = {
                1: 0.03,   # car
                2: 0.03,   # truck
                3: 0.03,   # bus
                4: 0.03,   # trailer
                5: 0.02,   # construction vehicle
                6: 0.08,   # pedestrian 
                7: 0.02,   # motorcycle
                8: 0.02,   # bicycle
                9: 0.02,   # barrier
                10: 0.01   # traffic cone 
            }
            detections = []
            for i, obj in enumerate(msg.objects):
                label_str = obj.label.strip().lower()
                label = self.LABEL_STR_TO_ID.get(label_str, -1)

                if label not in VALID_CLASSES:
                    # rospy.logwarn(f"[SKIP] Unknown or ignored class '{obj.label}' → mapped id: {label}")
                    continue

                if obj.score < class_min_confidence.get(label, 0.0):
                    # rospy.loginfo(f"[SKIP] Low confidence {obj.score:.3f} for class '{obj.label}'")
                    continue

                det = {
                    "id":           i,
                    "position":     [obj.x, obj.y, obj.z],  # ← 기존 [obj.x, obj.y] → z 추가
                    "yaw":          obj.yaw,
                    "size":         [obj.l, obj.w, obj.h],
                    "type":         label,
                    "reproj_bbox":  obj.boundingbox if len(obj.boundingbox) == 4 else [0.0] * 4,
                    "velocity":     [obj.vx, obj.vy],
                    "confidence":   obj.score
                }
                detections.append(det)

            rospy.logdebug(f"[DEBUG] 변환된 detections: count={len(detections)}, ids={[d['id'] for d in detections]}")

            # 4) Ego 상태 업데이트 #자차부정확한경우ㅠ
            # self.ego_vel = msg.ego_vel_x
            # self.ego_yaw_rate = msg.ego_yaw_rate

            # 5) 트래커 업데이트
            if dt > 0:
                self.tracker.update(detections, dt,
                            ego_vel=self.ego_vel,
                            ego_yaw_rate=self.ego_yaw_rate,
                            ego_yaw=self.ego_yaw)
            else:
                rospy.logwarn(f"Skipping KF predict/update for dt={dt:.3f}s")

            # 6) 트랙 퍼블리시
            tracks = self.tracker.get_tracks()
            rospy.loginfo(f"[Tracker] Published Tracks: {len(tracks)}")

            ta = PfGMFATrackArray(header=msg.header)
            for t in tracks:
                m = PfGMFATrack()
                m.pos_x          = t["x"]
                m.pos_y          = t["y"]
                m.yaw            = t["yaw"]
                dims             = list(t["size"])[:3]
                m.boundingbox    = dims + [0.0] * 5
                m.confidence_ind = t["confidence"]
                m.id             = int(t["id"])
                m.obj_class      = t["type"]
                ta.tracks.append(m)
            # self.tracking_pub.publish(ta)

            # === [7] RViz 마커 시각화 ===
            vis_header = Header(frame_id="vehicle", stamp=msg.header.stamp)

            # 1. 현재 ID 모음
            current_ids = set(t["id"] for t in tracks)

            # 2. 이전에 있었지만 지금은 없는 → 삭제 대상
            deleted_ids = getattr(self, "prev_track_ids", set()) - current_ids

            # 3. 새 마커 array로 초기화
            self.marker_array = MarkerArray()

            # 4. 삭제 마커 추가
            for tid in deleted_ids:
                delete_marker = Marker()
                delete_marker.header = vis_header
                delete_marker.ns = "track_meshes"
                delete_marker.id = tid
                delete_marker.action = Marker.DELETE
                self.marker_array.markers.append(delete_marker)

                delete_text = Marker()
                delete_text.header = vis_header
                delete_text.ns = "track_ids"
                delete_text.id = 1000 + tid
                delete_text.action = Marker.DELETE
                self.marker_array.markers.append(delete_text)

                delete_arrow = Marker()
                delete_arrow.header = vis_header
                delete_arrow.ns = "track_arrows"
                delete_arrow.id = 2000 + tid
                delete_arrow.action = Marker.DELETE
                self.marker_array.markers.append(delete_arrow)



            # 5. 현재 트랙 마커 추가
            for t in tracks:
                marker = create_single_track_marker(t, vis_header, t["id"])
                self.marker_array.markers.append(marker)

                text_marker = create_text_marker(t, vis_header, 1000 + t["id"])
                self.marker_array.markers.append(text_marker)

                # ✅ [New] 속도 방향 화살표 추가
                vx, vy = t.get("velocity", [0.0, 0.0]) if "velocity" in t else [0.0, 0.0]
                dx = vx
                dy = vy

                arrow = Marker()
                arrow.header = vis_header
                arrow.ns = "track_arrows"
                arrow.id = 2000 + t["id"]
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.scale.x = 0.2
                arrow.scale.y = 0.5
                arrow.scale.z = 0.3
                arrow.color.a = 1.0
                arrow.color.r = 1.0
                arrow.color.g = 1.0
                arrow.color.b = 1.0  # 흰색

                z_base = t["position"][2] if "position" in t and len(t["position"]) > 2 else 0.0
                z_center = z_base + t["size"][2] / 2.0

                arrow.points.append(Point(
                    x=t["x"],
                    y=t["y"],
                    z=z_center
                ))
                arrow.points.append(Point(
                    x=t["x"] + dx,
                    y=t["y"] + dy,
                    z=z_center
                ))

                self.marker_array.markers.append(arrow)
    
            # 6. 자차 마커 추가
            ego_marker = create_ego_marker(vis_header.stamp)
            self.marker_array.markers.append(ego_marker)

            # 7. 퍼블리시
            self.vis_pub.publish(self.marker_array)

            # 8. 트랙 ID 갱신
            self.prev_track_ids = current_ids
        except Exception as e:
            rospy.logerr(f"[detection_callback] Unexpected error: {e}\n{traceback.format_exc()}")
            
if __name__ == '__main__':
    open("/tmp/mctrack_cost_debug.txt", "w").close() 
    try:
        MCTrackTrackerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 