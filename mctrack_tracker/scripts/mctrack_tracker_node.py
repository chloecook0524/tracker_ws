#!/usr/bin/env python3
# 표준 라이브러리
import uuid
import json
import math
import traceback
from collections import deque

# 타입 힌트
from typing import List, Dict

# 외부 라이브러리
import numpy as np
import cv2
from shapely.geometry import Polygon
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment
from lap import lapjv

# ROS 관련
import rospy
import tf
import tf.transformations
from std_msgs.msg import Header, Float32
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# 메시지 정의 (커스텀 포함)
from lidar_processing_msgs.msg import LidarPerceptionOutput, LidarObject
from lidar_processing_msgs.msg import PfGMFATrack, PfGMFATrackArray
from vdcl_fusion_perception.msg import DetectionObjects
from chassis_msgs.msg import Chassis

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

        # 초기 상태를 그대로 복사해서 fusion/predict용으로 사용
        self.global_velocity_fusion = self.global_velocity
        self.global_acceleration_fusion = self.global_acceleration
        self.global_yaw_fusion = self.global_yaw
        self.lwh_fusion = self.lwh

        # 속도 변화 추정 (단순 초기화)
        self.global_velocity_diff = [0, 0]
        self.global_velocity_curve = [0, 0]

        # 위치 + 크기 + 각도 통합 표현
        self.global_xyz_lwh_yaw = self.global_xyz + list(self.lwh) + [self.global_yaw]

        # 과거 위치 추정 (보간 가능)
        self.global_xyz_last = self.backward_prediction()
        self.global_xyz_lwh_yaw_last = self.global_xyz_last + list(self.lwh) + [self.global_yaw]

        self.global_xyz_lwh_yaw_predict = self.global_xyz_lwh_yaw
        self.global_xyz_lwh_yaw_fusion = self.global_xyz_lwh_yaw
        self.unmatch_length = 0

    def backward_prediction(self):
        last_xy = np.array(self.global_xyz[:2]) - np.array(self.global_velocity) * 0.5
        return last_xy.tolist() + [self.global_xyz[2]]
    

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

def get_class_weights(class_id):
    """
    클래스별 Hybrid Cost 계산 시 사용하는 GDIoU 중심 가중치
    - 반환값: (area_penalty_weight, center_dist_weight)
    """
    weights = {
        1: (0.5, 1.5),  # car
        2: (1.0, 1.0),  # truck
        3: (0.6, 0.4),  # bus
        4: (1.0, 1.0),  # trailer
        6: (1.0, 1.0),  # pedestrian
        7: (1.0, 1.0),  # motorcycle
        8: (1.0, 1.0),  # bicycle
        # 필요 시 다른 클래스 추가
    }
    return weights.get(class_id, (0.7, 0.3))  # default weight

def get_confidence_weights(class_id):
    """
    클래스별 confidence blending 가중치
    - 반환값: (det_score_weight, pred_score_weight)
    """
    weights = {
        1:  (0.5, 0.5),  # car
        2:  (0.5, 0.5),  # truck
        3:  (0.5, 0.5),  # bus
        4:  (0.6, 0.4),  # trailer
        5:  (0.6, 0.4),  # construction vehicle
        6:  (0.6, 0.4),  # pedestrian
        7:  (0.5, 0.5),  # motorcycle
        8:  (0.6, 0.4),  # bicycle
        9:  (0.8, 0.2),  # barrier
        10: (0.8, 0.2),  # traffic cone
    }
    return weights.get(class_id, (0.5, 0.5))

# === Utility Functions ===

def orientation_similarity(angle1_rad, angle2_rad):
    cosine_similarity = math.cos((angle1_rad - angle2_rad + np.pi) % (2 * np.pi) - np.pi)
    return (cosine_similarity + 1.0) / 2.0

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

    marker.pose.position.x = 1.5
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
    z_base = track["position"][2]
    m.pose.position.z = z_base

    
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
    # 🎯 특정 클래스는 ID 마커 생략
    if track["type"] in [9, 10]:  # 9: barrier, 10: traffic cone
        return None
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

def create_arrow_marker(track, header, marker_id):
    vx, vy = track.get("velocity", [0.0, 0.0])
    speed = math.hypot(vx, vy)

    arrow = Marker()
    arrow.header = header
    arrow.ns = "track_arrows"
    arrow.id = marker_id
    arrow.type = Marker.ARROW
    arrow.action = Marker.ADD
    arrow.scale.x = 0.2
    arrow.scale.y = 0.5
    arrow.scale.z = 0.3
    arrow.color.a = 1.0 if speed > 0.1 else 0.0
    arrow.color.r = 1.0
    arrow.color.g = 1.0
    arrow.color.b = 1.0

    z_base = track["position"][2]
    z_center = z_base + track["size"][2] / 2.0
    if track["type"] == 3:  # 🟡 bus
        z_center += 5.0

    arrow.points.append(Point(x=track["x"], y=track["y"], z=z_center))
    arrow.points.append(Point(x=track["x"] + vx, y=track["y"] + vy, z=z_center))

    return arrow

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


# === Hungarian IoU Matching Function with predicted boxes and distance-based cost ===
def hungarian_iou_matching(tracks, detections, dt=0.1, use_precise_gdiou=False):
    if not tracks or not detections:
        rospy.logwarn("[Hungarian Matching] No tracks or detections available!")
        return [], list(range(len(detections))), list(range(len(tracks))), [], []

    VALID_CLASS_IDS = set(CLASS_CONFIG.keys())
    detections = [d for d in detections if d["type"] in VALID_CLASS_IDS]

    cost_thresholds = {
        1: 1.8,   # car
        2: 2.0,   # truck
        3: 2.0,   # bus
        4: 2.0,   # trailer
        5: 2.0,   # construction_vehicle
        6: 2.0,   # pedestrian
        7: 2.0,   # motorcycle
        8: 2.0,   # bicycle
        9: 1.8,   # barrier
        10: 2.0,  # traffic cone
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

            if use_precise_gdiou:
                bbox_det = BBox(
                    frame_id=det.get("id", -1),
                    bbox={
                        "category": det["type"],
                        "detection_score": det.get("confidence", 0.5),
                        "lwh": det["size"],
                        "global_xyz": det["position"],
                        "global_orientation": [0, 0, 0, 1],
                        "global_yaw": det["yaw"],
                        "global_velocity": det.get("velocity", [0.0, 0.0]),
                        "global_acceleration": [0.0, 0.0],
                    }
                )
                ro_iou = cal_rotation_gdiou_inbev(track, bbox_det, class_id=track.label, cal_flag="Predict")
            else:
                ro_iou = ro_gdiou_2d(track.size[:2], det["size"][:2], track.x[3], det["yaw"])
            iou_penalty = 1.0 - max(ro_iou, 0.0)

            if track.label in [6, 7, 8]:  # 작은 객체
                cost = 0.5 * dist + 0.3 * yaw_penalty + 0.2 * iou_penalty
            else:
                cost = 0.4 * dist + 0.3 * yaw_penalty + 0.3 * iou_penalty
            # if track.label in [6, 7, 8]:
            #     with open("/tmp/mctrack_cost_debug.txt", "a") as f:
            #         f.write(
            #             f"[HYBRID_COST][CLS={track.label}] T#{i} ID={track.id} status={track.status_flag} "
            #             f"traj_len={track.traj_length} missed={track.missed_count} → "
            #             f"dist={dist:.2f}, yaw_diff={yaw_diff:.2f}, ro_gdiou={ro_iou:.3f}, cost={cost:.3f}\n"
            #         )

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

        if cost < threshold:
            matches.append((i, j))
            unmatched_tracks.discard(i)
            unmatched_dets.discard(j)
        # else:
        #     if label in [6, 7, 8]:  # 소형 객체만 로깅
        #         log_line = (f"[HUNGARIAN FAIL] CLS={label} T#{i} ID={tracks[i].id} vs D#{j} "
        #                     f"→ dist={np.linalg.norm(tracks[i].x[:2] - np.array(detections[j]['position'][:2])):.2f}, "
        #                     f"yaw_diff={abs(tracks[i].x[3] - detections[j]['yaw']):.2f}, "
        #                     f"cost={cost:.3f}, threshold={threshold:.2f}, ro_gdiou={ro_iou:.3f}")
        #         with open("/tmp/mctrack_cost_debug.txt", "a") as f:
        #             f.write(log_line + "\n")

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
    DEAD = 3

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
        'max_predict_time': 2.5, 
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
        'max_predict_time': 2.5, 
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
        'max_predict_time': 2.5, 
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
        'max_predict_time': 3.0, 
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
        'max_predict_time': 2.8, 
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
        'max_predict_time': 2.0, 
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
        'max_predict_time': 3.0, 
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
        'max_predict_time': 1.5, 
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
        'max_predict_time': 1.5, 
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
        'max_predict_time': 2.0, 
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

# === KalmanTrackedObject (1/2) ===
class KalmanTrackedObject:
    def __init__(self, detection, obj_id=None):
        self.id = obj_id or (uuid.uuid4().int & 0xFFFF)
        self.label = detection['type']
        wlh = detection.get('size', [1.0, 1.0, 1.0])
        self.traj_length = 1
        self.status_flag = TrackState.INITIALIZATION 
        self.hits = 1
        self.yaw_drift_buffer = deque(maxlen=5)
        px, py = detection['position'][:2]
        vx, vy = detection.get("velocity", [0.0, 0.0])
        self.pose_state = np.array([px, py, vx, vy])
        self.confidence = 0.0
        self.time_since_update = 0.0

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
        self.bboxes = []

    def predict(self, dt):
        # === Kalman Prediction (pose, velocity, yaw, size) ===
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        self.pose_state = F @ self.pose_state
        # ✅ 여기! 감쇠 적용
        self.pose_state[2:] *= 0.7  # 감쇠율은 0.6~0.9 사이 튜닝 가능

        self.pose_P = F @ self.pose_P @ F.T + self.pose_Q

        Fy = np.eye(2)
        Fy[0, 1] = dt
        self.yaw_state = Fy @ self.yaw_state
        self.yaw_P = Fy @ self.yaw_P @ Fy.T + self.yaw_Q
        self.yaw_state[0] = np.arctan2(np.sin(self.yaw_state[0]), np.cos(self.yaw_state[0]))

        self.size_P = self.size_P + self.size_Q

        self.age += dt
        self.time_since_update += dt

        self.confidence = self.tracking_score()


        pred_dict = {
            "frame_id": -1,
            "category": self.label,
            "detection_score": self.confidence,
            "lwh": self.size.tolist(),
            "global_xyz": self.pose_state[:2].tolist() + [self.bboxes[-1].global_xyz[2] if self.bboxes else 0.0],
            "global_orientation": [0, 0, 0, 1],
            "global_yaw": self.yaw_state[0],
            "global_velocity": self.pose_state[2:].tolist(),
            "global_acceleration": [0.0, 0.0],
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
            # === 클래스 설정 기반 삭제 유예 범위 계산 ===
            base_time = CLASS_CONFIG[self.label].get("max_predict_time", 2.5)
            max_extension = min(base_time * 1.5, 3.0)  # 최대 3초까지만 보류
            threshold = base_time + max_extension

            if self.traj_length > 10 and getattr(self, 'confidence', 0.5) > 0.6 and self.time_since_update < threshold:
                return

            self.status_flag = TrackState.DEAD

    def update(self, detection, dt):
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
        alpha = 0.5
        self.pose_state[2:] = alpha * self.pose_state[2:] + (1 - alpha) * np.array(vel)

        # ✅ 불확실도 리셋 (선택 사항)
        # self.pose_P[2:, 2:] = np.eye(2) * 1e-1

        # # ✅ Yaw 덮어쓰기 (간단화)
        # self.yaw_state[0] = detection['yaw']
        # self.yaw_state[1] = 0.0
        # self.yaw_P = np.eye(2) * 1e-2

        # === Yaw 제한 보정 with 튐 누적 보완 + 속도 기반 완화 ===
        def normalize_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        yaw_det = detection['yaw']
        yaw_diff = normalize_angle(yaw_det - self.yaw_state[0])
        self.yaw_drift_buffer.append(yaw_diff)
        if len(self.yaw_drift_buffer) > 5:
            self.yaw_drift_buffer.popleft()

        mean_drift = np.mean(self.yaw_drift_buffer)
        v = np.linalg.norm(self.pose_state[2:4])
        yaw_drift = abs(mean_drift)

        # === 조건별 보정 계수 설정
        if v < 0.2:
            coeff = 0.4  # 정지 시에도 빠르게 보정
        elif v < 1.0:
            coeff = 0.3
        else:
            coeff = 0.2 if yaw_drift < np.radians(3) else 0.4

        # === 보정 적용
        if yaw_drift > np.radians(15):
            # 🎯 너무 큰 튐은 detection 덮어쓰기
            self.yaw_state[0] = yaw_det
            self.yaw_state[1] = 0.0
            self.yaw_P = np.eye(2) * 1e-2
            self.yaw_drift_buffer.clear()
        elif abs(mean_drift) < 0.5:
            # 🔄 점진적 Kalman-like 보정
            self.yaw_state[0] += coeff * mean_drift
            self.yaw_state[0] = normalize_angle(self.yaw_state[0])
            self.yaw_state[1] = 0.0
            self.yaw_P = np.eye(2) * 1e-2
        elif len(self.yaw_drift_buffer) == 5 and all(abs(d) > 0.5 for d in self.yaw_drift_buffer):
            # 🧠 반복적으로 틀리면 강제 덮어쓰기
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
        self.missed_count = 0
        self.hits += 1
        self.time_since_update = 0.0
        self.traj_length += 1
        det_score = detection.get("confidence", 0.5)
        pred_score = self.tracking_score()

        alpha, beta = get_confidence_weights(self.label)
        self.confidence = alpha * det_score + beta * pred_score

        if self.traj_length > self.confirm_threshold or (
            self.confidence > CLASS_CONFIG[self.label]["confirmed_det_score"] and
            self.confidence > CLASS_CONFIG[self.label]["confirmed_match_score"]
        ):
            self.status_flag = TrackState.CONFIRMED


        new_bbox = BBox(frame_id=detection.get("id", 0), bbox={
            "category": self.label,
            "detection_score": self.confidence,
            "lwh": detection["size"],
            "global_xyz": detection["position"][:3],
            "global_orientation": [0, 0, 0, 1],
            "global_yaw": detection["yaw"],
            "global_velocity": detection.get("velocity", [0.0, 0.0]),
            "global_acceleration": [0.0, 0.0],
        })
        self.bboxes.append(new_bbox)
        if len(self.bboxes) > 30:
            self.bboxes.pop(0)

    def tracking_score(self):
        vel = np.hypot(self.pose_state[2], self.pose_state[3])
        expected_vel = self.expected_velocity
        vel_consistency = np.exp(-abs(vel - expected_vel) / (expected_vel + 1e-3))
        
        traj_penalty = np.exp(-0.2 * self.traj_length)  # trajectory 길이 penalty
        age_bonus = min(1.0, 0.1 * self.age)            # age 10초 이상이면 1.0으로 saturate

        score = (self.hits / (self.age + 1e-3)) * vel_consistency * (1.0 - traj_penalty) * age_bonus
        return max(0.1, min(1.0, score))

    def apply_ego_compensation(self, dx, dy, dyaw):
        cos_yaw = np.cos(-dyaw)
        sin_yaw = np.sin(-dyaw)

        rel_x = self.pose_state[0]
        rel_y = self.pose_state[1]

        self.pose_state[0] = cos_yaw * rel_x - sin_yaw * rel_y - dx
        self.pose_state[1] = sin_yaw * rel_x + cos_yaw * rel_y - dy
        self.yaw_state[0] -= dyaw
        self.yaw_state[0] = np.arctan2(np.sin(self.yaw_state[0]), np.cos(self.yaw_state[0]))

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
    def __init__(self, use_precise_gdiou=False):
        self.tracks = []
        self.use_precise_gdiou = use_precise_gdiou 

    def predict(self, dt):
        for t in self.tracks:
            t.predict(dt)
    
    def apply_ego_compensation_to_all(self, dx, dy, dyaw):
        for track in self.tracks:
            track.apply_ego_compensation(dx, dy, dyaw)

    # === Modify Soft-deleted ReID with reproj_bbox filtering ===
    def _recover_obscured_tracks(self, unmatched_dets: List[int], detections: List[Dict], dt: float) -> List[int]:
        used = []

        for di in unmatched_dets:
            det = detections[di]
            label = det['type']

            if label not in CLASS_CONFIG:
                continue

            best_track = None
            best_score = -1e9

            for track in self.tracks:
                if track.label != label:
                    continue
                if track.status_flag != TrackState.OBSCURED:
                    continue
                if track.traj_length > CLASS_CONFIG[label]["max_predict_len"]:
                    continue

                dx = track.pose_state[0] - det["position"][0]
                dy = track.pose_state[1] - det["position"][1]
                dist = np.hypot(dx, dy)
                if dist > _get_class_distance_threshold(label):
                    continue

                # ✅ 선택적 GDIoU 계산 방식
                if self.use_precise_gdiou:
                    bbox_det = BBox(frame_id=det.get("id", -1), bbox={
                        "category": det["type"],
                        "detection_score": det.get("confidence", 0.5),
                        "lwh": det["size"],
                        "global_xyz": det["position"],
                        "global_orientation": [0, 0, 0, 1],
                        "global_yaw": det["yaw"],
                        "global_velocity": det.get("velocity", [0.0, 0.0]),
                        "global_acceleration": [0.0, 0.0],
                    })
                    score = cal_rotation_gdiou_inbev(track, bbox_det, class_id=label, cal_flag="BackPredict")
                else:
                    score = ro_gdiou_2d(track.size[:2], det['size'][:2],
                                        track.yaw_state[0], det['yaw'])

                cost = dist + (1.0 - score)

                if score < 0.5 and cost > 1.5:
                    continue

                if score > best_score:
                    best_score = score
                    best_track = track

            if best_track is not None:
                best_track.status_flag = TrackState.CONFIRMED
                confidence = det.get("confidence", 0.5)
                best_track.update(det, dt)
                best_track.hits += 1
                used.append(di)

        return used

    def update(self, detections, dt):
        # 0) Predict step for all existing tracks
        for t in self.tracks:
            t.predict(dt)

        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(self.tracks)))
        matched_ids = set()

        # 1) Matching using Hungarian algorithm
        if self.tracks and detections:
            _, unmatched_dets, unmatched_trks, matched_trks, matched_dets = \
                hungarian_iou_matching(
                    self.tracks,
                    detections,
                    dt=dt,
                    use_precise_gdiou=self.use_precise_gdiou
                )
            for tr, det in zip(matched_trks, matched_dets):
                tr.update(det, dt)
                matched_ids.add(tr.id)

        # 2) OBSCURED 처리: 짧은 트랙이 unmatch된 경우
        for ti in unmatched_trks:
            track = self.tracks[ti]
            if track.traj_length <= 1 and track.missed_count > 0:
                track.status_flag = TrackState.OBSCURED

        # 3~4) ReID로 OBSCURED 트랙 복구
        used_r = self._recover_obscured_tracks(unmatched_dets, detections, dt)
        unmatched_dets = [d for d in unmatched_dets if d not in used_r]

        # 5) 새로운 트랙 생성 (중복 방지 포함)
        for di in unmatched_dets:
            det = detections[di]
            is_duplicate = False
            for t in self.tracks:
                if t.label != det["type"]:
                    continue
                if t.status_flag not in [TrackState.CONFIRMED, TrackState.OBSCURED]:
                    continue
                dist = np.linalg.norm(t.x[:2] - det["position"][:2])
                if dist > 1.5:
                    continue
                score = ro_gdiou_2d(t.size[:2], det['size'][:2], t.x[3], det['yaw'])
                if score > 0.7:
                    is_duplicate = True
                    break
            if not is_duplicate:
                self.tracks.append(KalmanTrackedObject(det))

        # 6) 매칭되지 않은 트랙은 missed_count 증가
        for t in self.tracks:
            if t.id not in matched_ids:
                t.missed_count += 1

        # 7) Soft-delete 및 삭제 조건 판단
        MAX_EXTRA_MISSES_FOR_DELETE = 10
        for t in self.tracks:
            if t.status_flag == TrackState.CONFIRMED:
                if t.missed_count > t.max_missed + MAX_EXTRA_MISSES_FOR_DELETE:
                    t.status_flag = TrackState.OBSCURED
            else:
                if t.missed_count > t.max_missed:
                    t.status_flag = TrackState.OBSCURED

        # 9) 오래된 OBSCURED 트랙 제거
        self.tracks = [
            t for t in self.tracks
            if not (t.status_flag == TrackState.OBSCURED and t.time_since_update > 5.0)
        ]

        # 상태 출력
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
            if t.status_flag == TrackState.OBSCURED:
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

        # === [A] 타이머들 초기화 ===
        self.tracking_timer = rospy.Timer(rospy.Duration(0.1), self.tracking_publish_callback)  # 기존 퍼블리시 루프
        self.marker_timer   = rospy.Timer(rospy.Duration(0.1), self.visualization_timer_callback)  # RViz 마커용 루프
        # self.predict_timer  = rospy.Timer(rospy.Duration(0.1), self.tracker_loop)  # ✅ 트래커 예측 루프 (10Hz)

        self.last_predict_time = None  # ✅ 트래커 루프용 시간 변수

        self.frame_idx       = 0
        self.start_time      = rospy.Time.now()
        self.marker_array    = MarkerArray()
        self.prev_track_ids  = set()

        # 4) Kalman 트래커 초기화
        self.tracker = KalmanMultiObjectTracker(
            use_precise_gdiou=False  # ✅ 여기서 토글
        )

        # 5) 퍼블리셔 생성 & 구독자 연결 대기
        self.tracking_pub = rospy.Publisher("/tracking/objects",
                                            PfGMFATrackArray,
                                            queue_size=100)
        self.vis_pub = rospy.Publisher("/tracking/markers", MarkerArray, queue_size=10)

        # 6) 리플레이어 콜백 구독
        self.detection_sub = rospy.Subscriber("/detection_objects",
                                            DetectionObjects,
                                            self.detection_callback,
                                            queue_size=500,
                                            tcp_nodelay=True)
        self.chassis_sub = rospy.Subscriber("/chassis",
                                            Chassis,
                                            self.chassis_callback,
                                            queue_size=100)

        self.chassis_buffer = deque(maxlen=100)

        # 7) Ego 상태 변수
        self.ego_vel         = 0.0
        self.ego_yaw_rate    = 0.0
        self.ego_yaw         = 0.0
        self.last_time_stamp = None

        self.predict_timer = rospy.Timer(rospy.Duration(0.05), self.tracker_loop)  # 20Hz 루프 추가

        rospy.loginfo("MCTrackTrackerNode 초기화 완료.")

    def chassis_callback(self, msg):
        stamp_sec = msg.header.stamp.to_sec()  
        self.chassis_buffer.append((stamp_sec, msg))
    
    def get_nearest_chassis(self, target_stamp):
        if not self.chassis_buffer:
            return None
        times = [abs(t - target_stamp) for (t, _) in self.chassis_buffer]
        nearest_idx = int(np.argmin(times))
        return self.chassis_buffer[nearest_idx][1]

    def get_chassis_samples_in_range(self, t0: float, t1: float):
        return [
            msg for (ts, msg) in self.chassis_buffer
            if t0 <= ts <= t1
        ]    

    def integrate_ego_motion(self, samples, t0, t1, initial_yaw):
        dx_total, dy_total, dyaw_total = 0.0, 0.0, 0.0
        last_time = t0
        yaw = initial_yaw

        rear_axle_offset = 1.4 # ✅ 차량 중심에서 뒤차축까지의 거리 (차량마다 다름)

        for msg in samples:
            curr_time = msg.header.stamp.to_sec()
            dt = curr_time - last_time
            if dt <= 0:
                continue

            avg_speed_kph = (msg.whl_spd_fl + msg.whl_spd_fr +
                            msg.whl_spd_rl + msg.whl_spd_rr) / 4.0
            vel = avg_speed_kph / 3.6
            yaw_rate = msg.cr_yrs_yr * np.pi / 180.0

            # ✅ rear axle 기준으로 속도 보정
            v_rear = vel - yaw_rate * rear_axle_offset

            dx = v_rear * dt * np.cos(yaw)
            dy = v_rear * dt * np.sin(yaw)
            dyaw = yaw_rate * dt

            dx_total += dx
            dy_total += dy
            dyaw_total += dyaw
            yaw += dyaw

            last_time = curr_time

        return dx_total, dy_total, dyaw_total
    

    def visualization_timer_callback(self, event):
        # 10Hz 주기로 마커 퍼블리시만 담당
        if hasattr(self, "marker_array"):
            self.vis_pub.publish(self.marker_array)

    def tracking_publish_callback(self, event):
        if not hasattr(self, 'last_time_stamp'):
            return  # 아직 첫 detection이 안 들어왔으면 skip

        tracks = self.tracker.get_tracks()

        ta = PfGMFATrackArray()
        ta.header.stamp = rospy.Time.now() 
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

    def tracker_loop(self, event):
        now = rospy.Time.now()
        if self.last_predict_time is None:
            self.last_predict_time = now
            return

        dt = (now - self.last_predict_time).to_sec()
        self.last_predict_time = now

        if dt <= 0:
            return

        self.tracker.predict(dt)


    def delete_all_markers(self):
        for i, marker in reversed(list(enumerate(self.marker_array.markers))):
            if marker.action == Marker.DELETE:
                del self.marker_array.markers[i]
            else:
                marker.action = Marker.DELETE
        self.vis_pub.publish(self.marker_array)
        self.marker_array = MarkerArray()  


    def detection_callback(self, msg):
        try:
            # [1] 타임스탬프 계산 및 자차 상태 추정
            timestamp_sec = msg.header.stamp.to_sec()

            # ✅ 최초 예측 시간 초기화 (타이머 루프에서 predict할 때 사용)
            if self.last_predict_time is None:
                self.last_predict_time = msg.header.stamp

            # ✅ 자차 상태 추정 (속도 + 요레이트)
            nearest_chassis_msg = self.get_nearest_chassis(timestamp_sec)
            if nearest_chassis_msg:
                avg_speed_kph = (
                    nearest_chassis_msg.whl_spd_fl + nearest_chassis_msg.whl_spd_fr +
                    nearest_chassis_msg.whl_spd_rl + nearest_chassis_msg.whl_spd_rr
                ) / 4.0
                self.ego_vel = avg_speed_kph / 3.6
                self.ego_yaw_rate = nearest_chassis_msg.cr_yrs_yr * np.pi / 180.0

            # [2] 시간 간격 계산 (dt)
            if self.last_time_stamp is None:
                dt = 0.0
            else:
                dt = (msg.header.stamp - self.last_time_stamp).to_sec()
            self.last_time_stamp = msg.header.stamp

            self.frame_idx += 1
            rospy.loginfo(f"[Tracker] Frame {self.frame_idx} (dt={dt:.3f}s)")

            # [3] 동일 타임스탬프 중복 수신 방지
            if hasattr(self, "last_timestamp_sec") and abs(timestamp_sec - self.last_timestamp_sec) < 1e-6:
                rospy.logwarn(f"[WARN] 동일한 timestamp가 반복 수신됨: {timestamp_sec:.6f}")
            self.last_timestamp_sec = timestamp_sec

            # [4] Detection 메시지 → 내부 dict 포맷으로 변환 + 필터링
            VALID_CLASSES = set(CLASS_CONFIG.keys())
            class_min_confidence = {
                1: 0.03, 2: 0.04, 3: 0.03, 4: 0.03, 5: 0.02,
                6: 0.08, 7: 0.02, 8: 0.02, 9: 0.02, 10: 0.01
            }
            detections = []
            for i, obj in enumerate(msg.objects):
                label_str = obj.label.strip().lower()
                label = self.LABEL_STR_TO_ID.get(label_str, -1)
                if label not in VALID_CLASSES:
                    continue
                if obj.score < class_min_confidence.get(label, 0.0):
                    continue
                detections.append({
                    "id":         i,
                    "position":   [obj.x, obj.y, obj.z],
                    "yaw":        obj.yaw,
                    "size":       [obj.l, obj.w, obj.h],
                    "type":       label,
                    "velocity":   [obj.vx, obj.vy],
                    "confidence": obj.score
                })

            # [5] Ego-motion 보상만 수행 (predict는 별도 루프에서 처리됨)
            if dt > 0:
                t1 = timestamp_sec
                t0 = t1 - dt
                samples = self.get_chassis_samples_in_range(t0, t1)

                if samples:
                    # ✅ chassis 데이터를 기반으로 정확한 적분 방식 보상 적용
                    dx, dy, dyaw = self.integrate_ego_motion(samples, t0, t1, self.ego_yaw)
                    self.tracker.apply_ego_compensation_to_all(dx, dy, dyaw)
                else:
                    # ✅ chassis 데이터가 없으면 보상 생략 (추론 왜곡 방지)
                    rospy.logwarn("[EgoComp] No chassis samples available — skipping ego-motion compensation")
            else:
                rospy.logwarn(f"Skipping ego compensation for dt={dt:.3f}s")

            # [6] 트래커 업데이트만 수행 (예측은 타이머 루프에서 수행됨)
            self.tracker.update(detections, dt)

            # [7] 추적 결과를 변환하여 메시지로 구성
            tracks = self.tracker.get_tracks()
            rospy.loginfo(f"[Tracker] Published Tracks: {len(tracks)}")

            ta = PfGMFATrackArray(header=msg.header)
            for t in tracks:
                m = PfGMFATrack()
                m.pos_x = t["x"]
                m.pos_y = t["y"]
                m.yaw   = t["yaw"]
                m.boundingbox = list(t["size"])[:3] + [0.0] * 5
                m.confidence_ind = t["confidence"]
                m.id = int(t["id"])
                m.obj_class = t["type"]
                ta.tracks.append(m)

            # [8] RViz 마커 구성 및 삭제/갱신 처리
            vis_header = Header(frame_id="vehicle", stamp=msg.header.stamp)
            current_ids = set(t["id"] for t in tracks)
            deleted_ids = getattr(self, "prev_track_ids", set()) - current_ids
            self.marker_array = MarkerArray()

            for tid in deleted_ids:
                for ns, base_id in [("track_meshes", 0), ("track_ids", 1000), ("track_arrows", 2000)]:
                    m = Marker()
                    m.header = vis_header
                    m.ns = ns
                    m.id = base_id + tid
                    m.action = Marker.DELETE
                    self.marker_array.markers.append(m)

            for t in tracks:
                m1 = create_single_track_marker(t, vis_header, t["id"])
                self.marker_array.markers.append(m1)

                m2 = create_text_marker(t, vis_header, 1000 + t["id"])
                if m2 is not None:
                    self.marker_array.markers.append(m2)

                m3 = create_arrow_marker(t, vis_header, 2000 + t["id"])
                self.marker_array.markers.append(m3)

            ego_marker = create_ego_marker(vis_header.stamp)
            self.marker_array.markers.append(ego_marker)

            self.vis_pub.publish(self.marker_array)
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