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


# === Global Path to Baseversion Detection File ===
BASE_DET_JSON = "/home/chloe/SOTA/MCTrack/data/base_version/nuscenes/centerpoint/val.json"
GT_JSON_PATH = "/home/chloe/nuscenes_gt_valsplit.json"

# === Kalman Filter Configs for Pose, Size, and Yaw ===
KF_CONFIG = {
    "pose": {
        "F": lambda dt: np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]),
        "H": np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
        "Q": np.diag([0.5, 0.5, 1.0, 1.0]),
        "R": np.diag([0.5, 0.5]),
        "P": np.diag([1.0, 1.0, 10.0, 10.0])
    },
    "size": {
        "F": np.eye(3),
        "H": np.eye(3),
        "Q": np.diag([0.1, 0.1, 0.1]),
        "R": np.diag([0.1, 0.1, 0.1]),
        "P": np.eye(3)
    },
    "yaw": {
        "F": lambda dt: np.array([[1, dt], [0, 1]]),
        "H": np.eye(2),
        "Q": np.diag([0.1, 0.1]),
        "R": np.diag([0.2, 5.0]),
        "P": np.eye(2)
    }
}

def transform_3dbox2corners(center, size, yaw):
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
    # 클래스별 하이브리드 비중: (w1: ro_gdiou, w2: bev_gdiou)
    weights = {
        0: (0.7, 0.3),  # car
        1: (0.9, 0.1),  # pedestrian
        2: (0.8, 0.2),  # bicycle
        3: (0.8, 0.2),  # motorcycle
        4: (0.6, 0.4),  # bus
        5: (0.5, 0.5),  # trailer
        6: (0.6, 0.4),  # truck
    }
    return weights.get(class_id, (0.7, 0.3))

# def get_class_weights(class_id):
#     weights = {
#         0: (0.5, 1.5),  # car
#         1: (1.0, 1.0),  # pedestrian
#         2: (1.0, 1.0),  # bicycle
#         3: (1.0, 1.0),  # motorcycle
#         4: (1.0, 1.0),  # bus
#         5: (1.0, 1.0),  # trailer
#         6: (1.0, 1.0),  # truck
#     }
#     return weights.get(class_id, (1.0, 1.0))

def cal_rotation_gdiou_inbev(box1, box2, yaw1, yaw2, class_id, center1, center2):
    """
    하이브리드 코스트 함수 (Ro-GDIoU + BEV GDIoU).
    
    Parameters:
    - box1, box2: [w, l]
    - yaw1, yaw2: 각도 (radian)
    - class_id: 객체 클래스 (0~6)
    - center1, center2: 중심 좌표 [x, y]
    """
    # 1. BEV IoU 계산
    corners1 = transform_3dbox2corners(center1, box1, yaw1)
    corners2 = transform_3dbox2corners(center2, box2, yaw2)
    bev_iou = polygon_iou(corners1, corners2)

    # 2. yaw 차이에 대한 cosine penalty
    yaw_diff = abs(yaw1 - yaw2)
    yaw_penalty = 1.0 - np.cos(yaw_diff)

    # 3. BEV GDIoU 보정
    bev_gdiou = bev_iou - 0.1 * yaw_penalty
    bev_gdiou = max(0.0, bev_gdiou)

    # 4. 기존 Ro-GDIoU 계산
    ro_iou = ro_gdiou_2d(box1, box2, yaw1, yaw2)

    # 5. 클래스별 가중 평균
    w1, w2 = get_class_weights(class_id)
    hybrid_score = w1 * ro_iou + w2 * bev_gdiou
    return hybrid_score

# === Utility Functions ===
def compute_yaw_similarity(yaw1, yaw2):
    dyaw = abs(yaw1 - yaw2)
    return max(0.0, np.cos(dyaw)) 

def iou_2d(box1, box2):
    w1, l1 = box1[0], box1[1]
    w2, l2 = box2[0], box2[1]
    if w1 <= 0 or l1 <= 0 or w2 <= 0 or l2 <= 0:
        return 0.0
    inter_w = min(w1, w2)
    inter_l = min(l1, l2)
    inter_area = inter_w * inter_l
    union_area = w1 * l1 + w2 * l2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

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

def create_tracking_markers(tracks, header):
    markers = MarkerArray()
    for i, t in enumerate(tracks):
        # box (cube)
        m = Marker()
        m.header = header
        m.ns = "track_boxes"
        m.id = i
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = t["x"]
        m.pose.position.y = t["y"]
        m.pose.position.z = 1.0  # 중간 높이

        from tf.transformations import quaternion_from_euler
        q = quaternion_from_euler(0, 0, t["yaw"])
        m.pose.orientation.x = q[0]
        m.pose.orientation.y = q[1]
        m.pose.orientation.z = q[2]
        m.pose.orientation.w = q[3]

        m.scale.x = t["size"][0]
        m.scale.y = t["size"][1]
        m.scale.z = t["size"][2] if len(t["size"]) > 2 else 1.5

        m.color.a = 0.5
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 0.0
        markers.markers.append(m)

        # text
        t_m = Marker()
        t_m.header = header
        t_m.ns = "track_ids"
        t_m.id = 1000 + i
        t_m.type = Marker.TEXT_VIEW_FACING
        t_m.action = Marker.ADD
        t_m.pose.position.x = t["x"]
        t_m.pose.position.y = t["y"]
        t_m.pose.position.z = 2.5
        t_m.scale.z = 0.8
        t_m.color.a = 1.0
        t_m.color.r = 1.0
        t_m.color.g = 1.0
        t_m.color.b = 1.0
        t_m.text = str(t["id"])
        markers.markers.append(t_m)

    return markers


def create_gt_markers(gt_tracks, header):
    markers = MarkerArray()
    for i, obj in enumerate(gt_tracks):
        m = Marker()
        m.header = header
        m.ns = "gt_boxes"
        m.id = 2000 + i
        m.type = Marker.CUBE
        m.action = Marker.ADD
        pos = obj.get('translation', [0,0,0])
        size = obj.get('size', [1,1,1])
        yaw = obj.get('rotation_yaw', 0.0) - np.pi/2

        m.pose.position.x = pos[0]
        m.pose.position.y = pos[1]
        m.pose.position.z = 1.0  # 높이 중간

        from tf.transformations import quaternion_from_euler
        q = quaternion_from_euler(0, 0, yaw)
        m.pose.orientation.x = q[0]
        m.pose.orientation.y = q[1]
        m.pose.orientation.z = q[2]
        m.pose.orientation.w = q[3]

        m.scale.x = size[0]
        m.scale.y = size[1]
        m.scale.z = size[2]

        m.color.a = 0.4
        m.color.r = 0.0
        m.color.g = 0.5
        m.color.b = 1.0

        markers.markers.append(m)
    return markers

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


# def _get_reproj_iou_thresh(label):
#     # From official MCTrack config: THRESHOLD.RV.COST_THRE
#     return -0.3

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
def hungarian_iou_matching(tracks, detections, use_hybrid_cost=False):
    if not tracks or not detections:
        rospy.logwarn("[Hungarian Matching] No tracks or detections available!")
        return [], list(range(len(detections))), list(range(len(tracks))), [], []

    # # 🛠️ 클래스별 cost threshold 설정 (트레일러만 tighter)
    # cost_thresholds = {
    #     4: 1.26,  # trailer
    # }
    # default_threshold = 2.2  # 나머지 클래스들은 모두 2.2 적용

    cost_thresholds = {
        0: 1.10,  # car
        1: 2.06,  # pedestrian
        2: 2.00,  # bicycle
        3: 2.06,  # motorcycle
        4: 1.60,  # bus
        5: 1.26,  # trailer
        6: 1.16,  # truck
    }
    default_threshold = 2.2

    cost_matrix = np.ones((len(tracks), len(detections)))

    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            box1 = track.size[:2]
            box2 = det["size"][:2]
            center1 = track.x[:2]
            center2 = det["position"]
            class_id = getattr(track, 'label', 0)

            dx = center1[0] - center2[0]
            dy = center1[1] - center2[1]
            dist = np.hypot(dx, dy)

            # ✂️ 하이브리드 방식에서만 pruning
            if use_hybrid_cost and dist > 10.0:
                continue

            if use_hybrid_cost:
                # ✳️ 하이브리드 방식 (정밀)
                iou_score = cal_rotation_gdiou_inbev(box1, box2, track.x[3], det["yaw"], class_id, center1, center2)
            else:
                # 🔁 기존 방식 (간단)
                iou_score = ro_gdiou_2d(box1, box2, track.x[3], det["yaw"])

            iou_cost = 1.0 - iou_score
            dist_cost = dist
            cost_matrix[i, j] = iou_cost + 0.5 * dist_cost

    # rospy.loginfo(f"Cost Matrix: {cost_matrix}")  

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, unmatched_tracks, unmatched_dets = [], set(range(len(tracks))), set(range(len(detections)))

    for r, c in zip(row_ind, col_ind):
        label = getattr(tracks[r], 'label', None)
        threshold = cost_thresholds.get(label, default_threshold)

        box1 = tracks[r].size[:2]
        box2 = detections[c]["size"][:2]
        ro_iou = ro_gdiou_2d(box1, box2, tracks[r].x[3], detections[c]["yaw"])

        if cost_matrix[r, c] < threshold and ro_iou > 0.1:
            matches.append((r, c))
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)
            
    # if matches:
    #     rospy.loginfo(f"[Hungarian Matching] Matched {len(matches)} tracks with detections.")
    # else:
    #     rospy.logwarn("[Hungarian Matching] No matches found.")

    matched_tracks = [tracks[r] for r, _ in matches]
    matched_detections = [detections[c] for _, c in matches]

    return matches, list(unmatched_dets), list(unmatched_tracks), matched_tracks, matched_detections

# === TrackState Enum ===
class TrackState:
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3

# === 클래스별 칼만 필터 설정 (MCTRACK 완성 버전) ===
CLASS_CONFIG = {
    0: {  # car
        "confirm_threshold": 2,
        "max_unmatch": 3,
        "expected_velocity": 10.0,
        "P": np.diag([1.0, 1.0, 10.0, 10.0]),
        "Q": np.diag([0.5, 0.5, 1.5, 1.5]),
        "R": np.diag([0.7, 0.7]),
        "P_size": np.diag([1.0, 1.0, 10.0]),
        "Q_size": np.diag([0.5, 0.5, 1.5]),
        "R_size": np.diag([2.0, 2.0, 2.0]),
        "P_yaw": np.diag([0.1, 0.1]),
        "Q_yaw": np.diag([0.1, 0.1]),
        "R_yaw": np.diag([0.2, 5.0]),
    },
    1: {  # pedestrian
        "confirm_threshold": 2,
        "max_unmatch": 3,
        "expected_velocity": 2.0,
        "P": np.diag([1.0, 1.0, 10.0, 10.0]),
        "Q": np.diag([1.5, 1.5, 1.5, 1.5]),
        "R": np.diag([2.0, 2.0]),
        "P_size": np.diag([1.0, 1.0, 10.0]),
        "Q_size": np.diag([0.5, 0.5, 1.5]),
        "R_size": np.diag([2.0, 2.0, 2.0]),
        "P_yaw": np.diag([0.1, 0.1]),
        "Q_yaw": np.diag([0.1, 0.1]),
        "R_yaw": np.diag([0.2, 5.0]),
    },
    2: {  # bicycle
        "confirm_threshold": 2,
        "max_unmatch": 3,
        "expected_velocity": 7.0,
        "P": np.diag([1.0, 1.0, 1.0, 1.0]),
        "Q": np.diag([0.3, 0.3, 1.0, 1.0]),
        "R": np.diag([0.1, 0.1]),
        "P_size": np.diag([1.0, 1.0, 10.0]),
        "Q_size": np.diag([0.5, 0.5, 1.5]),
        "R_size": np.diag([2.0, 2.0, 2.0]),
        "P_yaw": np.diag([0.1, 0.1]),
        "Q_yaw": np.diag([0.1, 0.1]),
        "R_yaw": np.diag([0.2, 5.0]),
    },
    3: {  # motorcycle
        "confirm_threshold": 2,
        "max_unmatch": 3,
        "expected_velocity": 8.0,
        "P": np.diag([1.0, 1.0, 10.0, 10.0]),
        "Q": np.diag([0.5, 0.5, 4.0, 4.0]),
        "R": np.diag([0.1, 0.1]),
        "P_size": np.diag([1.0, 1.0, 10.0]),
        "Q_size": np.diag([0.5, 0.5, 1.5]),
        "R_size": np.diag([2.0, 2.0, 2.0]),
        "P_yaw": np.diag([0.1, 0.1]),
        "Q_yaw": np.diag([0.1, 0.1]),
        "R_yaw": np.diag([0.2, 5.0]),
    },
    4: {  # bus
        "confirm_threshold": 2,
        "max_unmatch": 3,
        "expected_velocity": 6.0,
        "P": np.diag([100.0, 100.0, 100.0, 100.0]),
        "Q": np.diag([0.5, 0.5, 1.5, 1.5]),
        "R": np.diag([1.5, 1.5]),
        "P_size": np.diag([1.0, 1.0, 10.0]),
        "Q_size": np.diag([0.5, 0.5, 1.5]),
        "R_size": np.diag([2.0, 2.0, 2.0]),
        "P_yaw": np.diag([0.1, 0.1]),
        "Q_yaw": np.diag([0.1, 0.1]),
        "R_yaw": np.diag([0.2, 5.0]),
    },
    5: {  # trailer
        "confirm_threshold": 2,
        "max_unmatch": 3,
        "expected_velocity": 3.0,
        "P": np.diag([1.0, 1.0, 10.0, 10.0]),
        "Q": np.diag([0.3, 0.3, 0.1, 0.1]),
        "R": np.diag([2.0, 2.0]),
        "P_size": np.diag([1.0, 1.0, 10.0]),
        "Q_size": np.diag([0.5, 0.5, 1.5]),
        "R_size": np.diag([2.0, 2.0, 2.0]),
        "P_yaw": np.diag([0.1, 0.1]),
        "Q_yaw": np.diag([0.1, 0.1]),
        "R_yaw": np.diag([0.2, 5.0]),
    },
    6: {  # truck
        "confirm_threshold": 2,
        "max_unmatch": 3,
        "expected_velocity": 1.0,
        "P": np.diag([1.0, 1.0, 10.0, 10.0]),
        "Q": np.diag([0.1, 0.1, 2.0, 2.0]),
        "R": np.diag([1.5, 1.5]),
        "P_size": np.diag([1.0, 1.0, 10.0]),
        "Q_size": np.diag([0.5, 0.5, 1.5]),
        "R_size": np.diag([2.0, 2.0, 2.0]),
        "P_yaw": np.diag([0.1, 0.1]),
        "Q_yaw": np.diag([0.1, 0.1]),
        "R_yaw": np.diag([0.2, 5.0]),
    },
}


# === KalmanTrackedObject (1/2) ===
class KalmanTrackedObject:
    def __init__(self, detection, obj_id=None):
        self.id = obj_id or (uuid.uuid4().int & 0xFFFF)
        self.label = detection['type']
        wlh = detection.get('size', [1.0, 1.0, 1.0])
        self.reproj_bbox = detection.get('reproj_bbox')
        self.traj_length = 1
        self.use_smoothing = True  

        px, py = detection['position']
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
            self.yaw_R = class_cfg["R_yaw"]
            self.expected_velocity = class_cfg["expected_velocity"]
            self.confirm_threshold = class_cfg["confirm_threshold"]
            self.max_missed = class_cfg["max_unmatch"]
        else:
            self.pose_P = np.diag([1.0, 1.0, 10.0, 10.0])
            self.pose_Q = np.diag([0.5, 0.5, 1.5, 1.5])
            self.pose_R = np.diag([0.7, 0.7])
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
        self.hits = 1
        self.state = TrackState.CONFIRMED if self.hits >= self.confirm_threshold else TrackState.TENTATIVE
        self.soft_deleted = False

    def predict(self, dt, ego_vel=0.0, ego_yaw_rate=0.0, ego_yaw=0.0):
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        pred_pose_state = F @ self.pose_state
        pred_pose_P = F @ self.pose_P @ F.T + self.pose_Q

        if self.use_smoothing:
            # ✅ smoothing 적용
            alpha = 0.8
            self.pose_state[0:2] = pred_pose_state[0:2]
            self.pose_state[2] = alpha * self.pose_state[2] + (1-alpha) * pred_pose_state[2]
            self.pose_state[3] = alpha * self.pose_state[3] + (1-alpha) * pred_pose_state[3]
        else:
            # ✅ smoothing 없이 덮어쓰기
            self.pose_state = pred_pose_state

        self.pose_P = pred_pose_P

        # Yaw prediction
        Fy = np.eye(2)
        Fy[0, 1] = dt
        self.yaw_state = Fy @ self.yaw_state
        self.yaw_P = Fy @ self.yaw_P @ Fy.T + self.yaw_Q

        self.age += dt
        self.missed_count += 1

        # Normalize yaw
        self.yaw_state[0] = np.arctan2(np.sin(self.yaw_state[0]), np.cos(self.yaw_state[0]))

        # Size prediction: static
        self.size_P = self.size_P + self.size_Q

    def update(self, detection, dt):
        # Pose update
        z_pose = np.array([detection['position'][0], detection['position'][1]])
        H_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        y_pose = z_pose - H_pose @ self.pose_state
        S_pose = H_pose @ self.pose_P @ H_pose.T + self.pose_R[:2, :2]
        K_pose = self.pose_P @ H_pose.T @ np.linalg.inv(S_pose)
        self.pose_state += K_pose @ y_pose
        self.pose_P = (np.eye(4) - K_pose @ H_pose) @ self.pose_P

        # Velocity magnitude 업데이트
        vx, vy = self.pose_state[2], self.pose_state[3]
        speed = np.hypot(vx, vy)

        # Yaw update
        z_yaw = np.array([detection['yaw'], 0.0])
        H_yaw = np.array([[1, 0], [0, 1]])
        y_yaw = z_yaw - H_yaw @ self.yaw_state
        S_yaw = H_yaw @ self.yaw_P @ H_yaw.T + self.yaw_R
        K_yaw = self.yaw_P @ H_yaw.T @ np.linalg.inv(S_yaw)
        self.yaw_state += K_yaw @ y_yaw
        self.yaw_P = (np.eye(2) - K_yaw @ H_yaw) @ self.yaw_P

        # Normalize yaw
        self.yaw_state[0] = np.arctan2(np.sin(self.yaw_state[0]), np.cos(self.yaw_state[0]))

        # Reset flags
        self.soft_deleted = False
        self.missed_count = 0
        self.hits += 1
        self.traj_length += 1
        if self.hits >= self.confirm_threshold:
            self.state = TrackState.CONFIRMED

        # Size update
        z_size = np.array(detection['size'][:3])
        H_size = np.eye(3)
        y_size = z_size - H_size @ self.size_state
        S_size = H_size @ self.size_P @ H_size.T + self.size_R
        K_size = self.size_P @ H_size.T @ np.linalg.inv(S_size)
        self.size_state += K_size @ y_size
        self.size_P = (np.eye(3) - K_size @ H_size) @ self.size_P
        self.size_state = np.clip(self.size_state, a_min=0.1, a_max=20.0)    

        self.reproj_bbox = detection.get('reproj_bbox')

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
        """
        Try to resurrect (re-ID) any soft_deleted tracks by matching to
        remaining detections using a high-threshold Ro-GDIoU + reproj-IoU check.
        Returns the list of detection indices that were used to revive tracks.
        """
        used = []
        for di in unmatched_dets:
            det = detections[di]
            best_score = 0.0
            best_track = None
            for track in self.tracks:
                if not getattr(track, 'soft_deleted', False):
                    continue
                if det['type'] != track.label:
                    continue
                dx = track.pose_state[0] - det["position"][0]
                dy = track.pose_state[1] - det["position"][1]
                if np.hypot(dx, dy) > _get_class_distance_threshold(track.label):
                    continue

                score = ro_gdiou_2d(track.size[:2],
                                    det['size'][:2],
                                    track.yaw_state[0],
                                    det['yaw'])

                bbox1 = getattr(track, 'reproj_bbox', None)
                bbox2 = det.get('reproj_bbox', None)
                if bbox1 and bbox2 and bbox_iou_2d(bbox1, bbox2) < 0.1:
                    continue

                if score > best_score and score > 0.6:
                    best_score = score
                    best_track = track

            if best_track is not None:
                best_track.soft_deleted = False
                best_track.update(det, dt)
                best_track.hits += 1
                used.append(di)

        return used


    def _fallback_match(self,
                        unmatched_trks: List[int],
                        unmatched_dets: List[int],
                        detections: List[Dict],
                        dt: float) -> List[int]:
        """
        For each still-unmatched track, find the best remaining detection by
        Ro-GDIoU + distance, update that track, and return which detection indices
        were consumed.
        """
        used = []
        for ti in unmatched_trks:
            track = self.tracks[ti]
            best_score = -1.0
            best_det = -1
            for di in unmatched_dets:
                if di in used:
                    continue
                det = detections[di]
                if det['type'] != track.label:
                    continue
                dx = track.pose_state[0] - det["position"][0]
                dy = track.pose_state[1] - det["position"][1]
                if np.hypot(dx, dy) > _get_class_distance_threshold(track.label):
                    continue

                score = ro_gdiou_2d(track.size[:2],
                                    det['size'][:2],
                                    track.yaw_state[0],
                                    det['yaw'])
                if score > best_score and score > 0.3:
                    best_score = score
                    best_det = di

            if best_det >= 0:
                self.tracks[ti].update(detections[best_det], dt)
                used.append(best_det)

        return used


    def update(self, detections, dt, ego_vel=0.0, ego_yaw_rate=0.0, ego_yaw=0.0):
        # 0) predict
        for t in self.tracks:
            t.predict(dt, ego_vel, ego_yaw_rate, ego_yaw)

        # prepare unmatched lists
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(self.tracks)))

        # 1) Standard Hungarian matching
        if self.use_hungarian and self.tracks and detections:
            matches_h, unmatched_dets, unmatched_trks, matched_trks, matched_dets = \
                hungarian_iou_matching(self.tracks, detections, self.use_hybrid_cost)
            for tr, det in zip(matched_trks, matched_dets):
                tr.update(det, dt)

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
        for di in unmatched_dets:
            self.tracks.append(KalmanTrackedObject(detections[di]))

        # 6) Soft-delete vs Hard-delete
        for t in self.tracks:
            if t.missed_count > t.max_missed:
                t.soft_deleted = True
        # hard-delete buffer +10
        self.tracks = [
            t for t in self.tracks
            if not (t.soft_deleted and t.missed_count > t.max_missed + 10)
        ]

        rospy.loginfo(f"[Tracker] Total Tracks: {len(self.tracks)}")

    def get_tracks(self):
        results = []
        for t in self.tracks:
            if t.state != TrackState.CONFIRMED:
                continue
            if getattr(t, 'soft_deleted', False):
                continue
            if getattr(t, 'traj_length', 0) < 1:
                continue

            score = t.tracking_score()
            
            # rospy.loginfo(f"[Track Score] id={t.id}, score={score:.3f}")

            # ✅ 여기에 명확히 tracking_score 필터링 추가
            if score < 0.2:
                continue

            x, y, yaw = t.x[0], t.x[1], t.x[3]
            size = t.size

            results.append({
                "id":         t.id,
                "x":          x,
                "y":          y,
                "yaw":        yaw,
                "size":       size,
                "confidence": score,
                "type":       t.label
            })
        return results
        
class MCTrackTrackerNode:
    def __init__(self):
        # 1) 반드시 init_node 부터 호출
        rospy.init_node("mctrack_tracker_node", anonymous=True)

        # 2) logger_ready 파라미터가 올라올 때까지 대기 (최대 15초)
        rospy.loginfo("[Tracker] /logger_ready 기다리는 중…")
        start = rospy.Time.now()
        while not rospy.has_param("/logger_ready") and (rospy.Time.now() - start) < rospy.Duration(15.0):
            if rospy.is_shutdown():
                return
            rospy.sleep(0.1)
        if rospy.has_param("/logger_ready"):
            rospy.loginfo("[Tracker] /logger_ready 감지, 시작합니다")
        else:
            rospy.logwarn("[Tracker] /logger_ready 대기 타임아웃, 계속 진행")

        # 3) GT JSON 로드
        with open(GT_JSON_PATH, 'r') as f:
            raw = json.load(f)
        raw_results = raw.get("results", {})
        if isinstance(raw_results, dict):
            self.gt_data = raw_results
        else:
            self.gt_data = {}
            for ann in raw_results:
                token = ann.get("sample_token") or ann.get("token")
                if token:
                    self.gt_data.setdefault(token, []).append(ann)

        self.total_frames = len(self.gt_data)
        self.frame_idx    = 0
        self.start_time   = rospy.Time.now()

        # === Static TF (map → base_link) 퍼블리시 세팅 ===
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.publish_static_tf()

        # 4) Kalman 트래커 초기화
        use_rv_matching = rospy.get_param("~is_rv_matching", False)
        use_hybrid = rospy.get_param("~use_hybrid_cost", False)

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
        rospy.loginfo("[Tracker] /tracking/objects 구독자 기다리는 중…")
        while self.tracking_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("[Tracker] 구독자 연결 완료, 리플레이어 구독 시작")

        # 6) 리플레이어 콜백 구독
        self.detection_sub = rospy.Subscriber("/lidar_detection",
                                              LidarPerceptionOutput,
                                              self.detection_callback,
                                              queue_size= len(self.gt_data),
                                              tcp_nodelay=True)
        self.vel_sub     = rospy.Subscriber("/ego_vel_x", Float32, self.vel_callback, queue_size=1)
        self.yawrate_sub = rospy.Subscriber("/ego_yaw_rate", Float32, self.yawrate_callback, queue_size=1)
        self.yaw_sub     = rospy.Subscriber("/ego_yaw", Float32, self.yaw_callback, queue_size=1)

        # 7) ego state 초기화 & 이전 타임스탬프 변수
        self.ego_vel         = 0.0
        self.ego_yaw_rate    = 0.0
        self.ego_yaw         = 0.0
        self.last_time_stamp = None
        self.last_token = None

        rospy.loginfo("MCTrackTrackerNode 초기화 완료.")

    def vel_callback(self, msg):
        self.ego_vel = msg.data

    def yawrate_callback(self, msg):
        self.ego_yaw_rate = msg.data

    def yaw_callback(self, msg):
        self.ego_yaw = msg.data

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

        rospy.loginfo("🛰️ [Tracker] Static TF (map → base_link) published.")    

    def detection_callback(self, msg):
        token = msg.header.frame_id
        try:
            # 디버그 시작
            rospy.logdebug(f"[DEBUG] → detection_callback 시작 token={token}, 객체수={len(msg.objects)}")

            # 1) dt 계산
            if self.last_time_stamp is None:
                dt = 0.0
            else:
                dt = (msg.header.stamp - self.last_time_stamp).to_sec()
            self.last_time_stamp = msg.header.stamp

            # 프레임 인덱스 및 토큰 로그 (중복 증가 제거)
            self.frame_idx += 1
            rospy.loginfo(f"[Tracker] Frame {self.frame_idx}/{self.total_frames}: {token} (dt={dt:.3f}s)")

            # 토큰 연속성 체크
            if self.last_token == token:
                rospy.logwarn(f"[WARN] 토큰이 반복 수신됨: {token}")
            self.last_token = token

            # 2) detection 변환 (score 필터링)
            class_min_confidence = {
                0: 0.15, 1: 0.16, 2: 0.20, 3: 0.15,
                4: 0.16, 5: 0.17, 6: 0.0
            }
            detections = []
            for i, obj in enumerate(msg.objects):
                if obj.score < class_min_confidence.get(obj.label, 0.3):
                    continue
                det = {
                    "id":           i,
                    "position":    [obj.pos_x, obj.pos_y],
                    "yaw":          obj.yaw,
                    "size":         obj.size,
                    "type":         obj.label,
                    "reproj_bbox": obj.bbox_image,
                }
                # 빈 bbox 경고
                if det["reproj_bbox"] == [0,0,0,0]:
                    rospy.logwarn(f"[RV-Match][INVALID] Detection ID {i} has empty bbox_image")
                detections.append(det)
            rospy.logdebug(f"[DEBUG] → 변환된 detections: count={len(detections)}, ids={[d['id'] for d in detections]}")

            # 3) GT 트랙 정보 (시각화용)
            gt_tracks = self.gt_data.get(token, [])

            # 4) predict + update: dt>0 일 때만
            if dt > 0:
                self.tracker.update(detections, dt,
                                    ego_vel=self.ego_vel,
                                    ego_yaw_rate=self.ego_yaw_rate,
                                    ego_yaw=self.ego_yaw)
            else:
                rospy.logwarn(f"Skipping KF predict/update for dt={dt:.3f}s")

            # 5) 결과 퍼블리시
            tracks = self.tracker.get_tracks()
            rospy.loginfo(f"[Tracker] GT Tracks: {len(gt_tracks)}, Detected Tracks: {len(tracks)}")

            ta = PfGMFATrackArray(header=msg.header)
            for t in tracks:
                m = PfGMFATrack()
                m.pos_x          = t["x"]
                m.pos_y          = t["y"]
                m.yaw            = t["yaw"]
                dims             = list(t["size"])[:3]
                m.boundingbox    = dims + [0.0]*5
                m.confidence_ind = t["confidence"]
                m.id             = int(t["id"])
                m.obj_class      = t["type"]
                ta.tracks.append(m)
            self.tracking_pub.publish(ta)
            rospy.loginfo(f"[Tracker] Published {len(ta.tracks)} tracks")

            # 6) RViz 시각화
            vis_header = Header(frame_id="map", stamp=rospy.Time.now())
            self.vis_pub.publish(create_tracking_markers(tracks, vis_header))
            if gt_tracks:
                self.vis_pub.publish(create_gt_markers(gt_tracks, vis_header))

        except Exception as e:
            rospy.logerr(f"[detection_callback] Unexpected error: {e}\n{traceback.format_exc()}")

if __name__ == '__main__':
    try:
        MCTrackTrackerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 