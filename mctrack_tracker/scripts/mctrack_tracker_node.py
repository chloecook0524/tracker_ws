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


# === Global Path to Baseversion Detection File ===
BASE_DET_JSON = "/home/chloe/SOTA/MCTrack/data/base_version/nuscenes/centerpoint/val.json"
GT_JSON_PATH = "/home/chloe/nuscenes_gt_valsplit.json"

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
        yaw  = obj.get('rotation_yaw', 0.0)

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
        return 0.25
    elif label in [1, 2, 3, 4]:  # car, truck, bus, trailer
        return 0.35
    return 0.3

# === Reprojection Matching Function ===
def image_plane_matching(tracks, detections):
    matches = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_dets = set(range(len(detections)))

    for ti, track in enumerate(tracks):
        best_iou, best_di = -1.0, -1
        for di, det in enumerate(detections):
            if det['type'] != track.label:
                continue
            bbox1 = getattr(track, 'reproj_bbox', None)
            bbox2 = det.get('reproj_bbox', None)
            if bbox1 is None or bbox2 is None:
                continue

            dx = track.x - det["position"][0]
            dy = track.y - det["position"][1]
            dist = np.hypot(dx, dy)
            if dist > _get_class_distance_threshold(track.label):
                continue

            iou = bbox_iou_2d(bbox1, bbox2)
            threshold = _get_reproj_iou_thresh(track.label)
            if iou > best_iou and iou > threshold:
                best_iou = iou
                best_di = di
        if best_di >= 0:
            matches.append((ti, best_di))
            unmatched_tracks.discard(ti)
            unmatched_dets.discard(best_di)

    return matches, list(unmatched_dets), list(unmatched_tracks)

# === Hungarian IoU Matching Function with predicted boxes and distance-based cost ===
def hungarian_iou_matching(tracks, detections):
    if not tracks or not detections:
        return [], list(range(len(detections))), list(range(len(tracks))), [], []

    # 🛠️ 클래스별 cost threshold 설정 (트레일러만 tighter)
    cost_thresholds = {
        4: 1.26,  # trailer
    }
    default_threshold = 2.2  # 나머지 클래스들은 모두 2.2 적용

    cost_matrix = np.ones((len(tracks), len(detections)))

    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            box1 = track.size[:2]
            box2 = det["size"][:2]

            # 🟨 Ro_GDIoU 계산
            ro_iou = ro_gdiou_2d(box1, box2, track.x[3], det["yaw"])

            # 거리 계산
            dx = track.x[0] - det["position"][0]
            dy = track.x[1] - det["position"][1]
            dist = np.hypot(dx, dy)

            # 🟦 최종 cost
            distance_cost_weight = 0.5
            iou_cost = 1.0 - ro_iou
            dist_cost = dist
            total_cost = iou_cost + distance_cost_weight * dist_cost
            cost_matrix[i, j] = total_cost

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

    matched_tracks = [tracks[r] for r, _ in matches]
    matched_detections = [detections[c] for _, c in matches]

    return matches, list(unmatched_dets), list(unmatched_tracks), matched_tracks, matched_detections

# === TrackState Enum ===
class TrackState:
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3

# === 클래스별 칼만 필터 설정 (megvii style, 5x5 version) ===
CLASS_CONFIG = {
    1: {"confirm_threshold": 2, "max_unmatch": 3, 
        "Q": np.diag([0.5, 0.5, 1.5, 1.5, 0.01]),
        "R": np.diag([0.7, 0.7, 0.5]),  # ✅ 3x3
        "P": np.diag([1.0, 1.0, 10.0, 10.0, 1.0]),
        "expected_velocity": 10.0},
    2: {"confirm_threshold": 2, "max_unmatch": 3,
        "Q": np.diag([0.5, 0.5, 1.5, 1.5, 0.01]),
        "R": np.diag([2.0, 2.0, 3.5]),
        "P": np.diag([1.0, 1.0, 10.0, 10.0, 1.0]),
        "expected_velocity": 7.0},
    3: {"confirm_threshold": 2, "max_unmatch": 3,
        "Q": np.diag([0.5, 0.5, 4.0, 4.0, 0.01]),
        "R": np.diag([0.1, 0.1, 0.1]),
        "P": np.diag([1.0, 1.0, 10.0, 10.0, 1.0]),
        "expected_velocity": 8.0},
    4: {"confirm_threshold": 2, "max_unmatch": 3,
        "Q": np.diag([0.5, 0.5, 1.5, 1.5, 0.01]),
        "R": np.diag([1.5, 1.5, 500.0]),
        "P": np.diag([100.0, 100.0, 100.0, 100.0, 1.0]),
        "expected_velocity": 6.0},
    6: {
        "confirm_threshold": 2,
        "max_unmatch": 3,
        "Q": np.diag([0.3, 0.3, 0.4, 0.025, 0.025]),
        "R": np.diag([0.4, 0.4, 0.2]),
        "P": np.diag([1.0, 1.0, 1.0, 0.4, 0.4]),
        "expected_velocity": 1.0
    },
    7: {"confirm_threshold": 2, "max_unmatch": 3,
        "Q": np.diag([0.3, 0.3, 1.0, 1.0, 0.01]),
        "R": np.diag([0.1, 0.1, 1.0]),
        "P": np.diag([1.0, 1.0, 1.0, 1.0, 1.0]),
        "expected_velocity": 3.0},
    8: {"confirm_threshold": 2, "max_unmatch": 3,
        "Q": np.diag([0.3, 0.3, 1.0, 1.0, 0.01]),
        "R": np.diag([0.1, 0.1, 1.0]),
        "P": np.diag([1.0, 1.0, 1.0, 1.0, 1.0]),
        "expected_velocity": 2.5},
}

# === KalmanTrackedObject (1/2) ===
class KalmanTrackedObject:
    def __init__(self, detection, obj_id=None):
        # Unique ID
        self.id = obj_id or (uuid.uuid4().int & 0xFFFF)

        # Class label & size
        self.label = detection['type']
        self.size = detection['size']
        self.reproj_bbox = detection.get('reproj_bbox')
        
        # State vector [x, y, vx, yaw, yaw_rate]
        x0, y0 = detection['position']
        yaw0   = detection['yaw']
        self.x = np.array([x0, y0, 0.0, yaw0, 0.0], dtype=float)

        # Covariance, process & measurement noise (using class-specific values)
        cfg = CLASS_CONFIG.get(self.label, {
            "Q": np.diag([0.1, 0.1, 0.5, 0.01, 0.01]),
            "R": np.diag([0.5, 0.5, 0.1]),
            "P": np.diag([1.0, 1.0, 1.0, 0.5, 0.5]),
            "confirm_threshold": 2,
            "max_unmatch": 3,
            "expected_velocity": 5.0  # 기본값
        }) 
        self.expected_velocity = cfg["expected_velocity"]
        self.Q = cfg["Q"]
        self.R = cfg["R"]
        self.P = cfg["P"]
        

        # Measurement matrix H: z = [x, y, yaw]
        self.H = np.zeros((3, 5))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 3] = 1.0

        # Tracking metadata
        self.confirm_threshold = cfg["confirm_threshold"]
        self.max_missed        = cfg["max_unmatch"]
        self.age               = 0.0
        self.missed_count      = 0
        self.hits              = 1
        self.state             = TrackState.CONFIRMED if self.hits >= self.confirm_threshold else TrackState.TENTATIVE
        self.soft_deleted = False 

    def predict(self, dt, ego_vel=0.0, ego_yaw_rate=0.0, ego_yaw=0.0):
        # F matrix
        F = np.eye(5)
        F[0,2] = dt * np.cos(self.x[3])
        F[1,2] = dt * np.sin(self.x[3])
        F[3,4] = dt
        # predict
        self.x = F.dot(self.x)
        self.P = F.dot(self.P).dot(F.T) + self.Q
        # # ego‐motion compensation
        # dx   = -ego_vel * dt * np.cos(ego_yaw)
        # dy   = -ego_vel * dt * np.sin(ego_yaw)
        # dyaw = -ego_yaw_rate * dt
        # self.x[0] += dx; self.x[1] += dy; self.x[3] += dyaw
        # bookkeeping
        self.age += dt
        self.missed_count += 1
        # ego-motion 보정 후 yaw를 -π ~ π로 정규화
        self.x[3] = np.arctan2(np.sin(self.x[3]), np.cos(self.x[3]))

    def update(self, detection, dt):
        prev_x = self.x[0]
        prev_y = self.x[1]

        z = np.array([detection['position'][0],
                    detection['position'][1],
                    detection['yaw']], dtype=float)

        z_pred = self.H.dot(self.x)
        y      = z - z_pred
        S      = self.H.dot(self.P).dot(self.H.T) + self.R
        K      = self.P.dot(self.H.T).dot(np.linalg.inv(S))

        self.x = self.x + K.dot(y)
        I      = np.eye(5)
        self.P = (I - K.dot(self.H)).dot(self.P)

        # ✅ 속도 업데이트
        dx = self.x[0] - prev_x
        dy = self.x[1] - prev_y
        speed = np.hypot(dx, dy) / dt if dt > 1e-3 else 0.0
        self.x[2] = speed

        # ✅ soft-delete 복구
        self.soft_deleted = False

        self.missed_count = 0
        self.hits += 1
        if self.hits >= self.confirm_threshold:
            self.state = TrackState.CONFIRMED

        self.reproj_bbox = detection.get('reproj_bbox')

    def tracking_score(self):
        age_decay = np.exp(-0.1 * self.age)
        vx = self.x[2]
        vel_consistency = np.exp(-abs(vx - self.expected_velocity) / self.expected_velocity)
        raw = (self.hits / (self.age + 1e-3)) * age_decay * vel_consistency
        return max(0.1, min(1.0, raw))

# === KalmanMultiObjectTracker (predict only) ===
class KalmanMultiObjectTracker:
    def __init__(self, use_hungarian=True):
        self.tracks = []
        self.use_hungarian = use_hungarian

    def predict(self, dt, ego_vel, ego_yaw_rate, ego_yaw):
        for t in self.tracks:
            t.predict(dt, ego_vel, ego_yaw_rate, ego_yaw)

    # === Modify Soft-deleted ReID with reproj_bbox filtering ===
    def _reid_soft_deleted_tracks(self, detections, dt):
        used_indices = set()
        for det_idx, det in enumerate(detections):
            best_score, best_track = 0.0, None
            for track in self.tracks:
                if not track.soft_deleted:
                    continue
                dx = track.x - det["position"][0]
                dy = track.y - det["position"][1]
                dist = np.hypot(dx, dy)
                if dist > _get_class_distance_threshold(track.label):
                    continue
                score = ro_gdiou_2d(track.size[:2], det['size'][:2], track.x[3], det['yaw'])

                # Additional reproj_bbox condition for conservative match
                bbox1 = getattr(track, 'reproj_bbox', None)
                bbox2 = det.get('reproj_bbox', None)
                if bbox1 is not None and bbox2 is not None:
                    reproj_iou = bbox_iou_2d(bbox1, bbox2)
                    if reproj_iou < 0.1:
                        continue

                if score > best_score and score > 0.6:
                    best_score = score
                    best_track = track
            if best_track and det_idx not in used_indices:
                best_track.soft_deleted = False
                best_track.update(det, dt)
                best_track.hits += 1
                used_indices.add(det_idx)


    def _fallback_match(self, unmatched_tracks, unmatched_dets, detections, dt):
        used_dets = set()
        for ti in unmatched_tracks:
            track = self.tracks[ti]
            best_score = -1.0
            best_det = -1
            for di in unmatched_dets:
                if di in used_dets:
                    continue
                det = detections[di]
                if det['type'] != track.label:
                    continue
                dist = np.hypot(track.x - det['position'][0], track.y - det['position'][1])
                if dist > _get_class_distance_threshold(track.label):
                    continue
                score = ro_gdiou_2d(track.size[:2], det['size'][:2], track.x[3], det['yaw'])
                if score > best_score and score > 0.3:
                    best_score = score
                    best_det = di
            if best_det >= 0:
                track.update(detections[best_det], dt)
                used_dets.add(best_det)


    def update(self, detections, dt):
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(self.tracks)))
        matches = []

        # ✅ 통일된 Hungarian Matching (Ro_GDIoU + 거리 기반)
        if self.use_hungarian and self.tracks and detections:
            matches, unmatched_dets, unmatched_trks, matched_tracks, matched_detections = \
                hungarian_iou_matching(self.tracks, detections)

            for tr, det in zip(matched_tracks, matched_detections):
                tr.update(det, dt)

        # 1️⃣ 보조 매칭 (fallback)
        self._fallback_match(unmatched_trks, unmatched_dets, detections, dt)

        # 2️⃣ 이미지 기반 보조 매칭
        image_matches, new_unmatched_dets, new_unmatched_trks = image_plane_matching(
            [self.tracks[i] for i in unmatched_trks],
            [detections[i] for i in unmatched_dets]
        )
        unmatched_trks = [unmatched_trks[i] for i in new_unmatched_trks]
        unmatched_dets = [unmatched_dets[i] for i in new_unmatched_dets]

        # 3️⃣ Soft-delete ReID
        self._reid_soft_deleted_tracks(detections, dt)

        for rel_trk_idx, rel_det_idx in image_matches:
            abs_trk_idx = unmatched_trks[rel_trk_idx]
            abs_det_idx = unmatched_dets[rel_det_idx]
            self.tracks[abs_trk_idx].update(detections[abs_det_idx], dt)

        # 4️⃣ New track 생성
        for di in unmatched_dets:
            self.tracks.append(KalmanTrackedObject(detections[di]))

        # 5️⃣ 죽은 트랙 제거
        for t in self.tracks:
            score = t.tracking_score()
            if t.missed_count > t.max_missed or score < 0.3:
                if not t.soft_deleted:
                    rospy.loginfo(f"[MCTrack][SoftDelete] Track ID {t.id} soft-deleted (missed={t.missed_count}, score={score:.2f})")
                t.soft_deleted = True
        
    def get_tracks(self):
        results = []
        for t in self.tracks:
            if t.state != TrackState.CONFIRMED:
                continue
            if getattr(t, 'soft_deleted', False):  # ✅ soft-deleted 상태는 출력 대상에서 제외
                continue

            score = t.tracking_score()
            if getattr(self, 'use_confidence_filtering', False) and score < 0.6:
                continue

            # 위치·yaw·크기 꺼내기
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
        self.tracker = KalmanMultiObjectTracker(use_hungarian=True)
        self.tracker.use_confidence_filtering = True

        # 5) 퍼블리셔 생성 & 구독자 연결 대기
        self.tracking_pub = rospy.Publisher("/tracking/objects",
                                            PfGMFATrackArray,
                                            queue_size=10)
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
        # 1) header.stamp 을 직접 쓰도록 dt 계산
        if self.last_time_stamp is None:
            dt = 0.0
        else:
            dt = (msg.header.stamp - self.last_time_stamp).to_sec()
        self.last_time_stamp = msg.header.stamp

        self.frame_idx += 1
        token = msg.header.frame_id
        rospy.loginfo(f"[Tracker] Frame {self.frame_idx}/{self.total_frames}: {token} (dt={dt:.3f}s)")

        # 2) msg.objects → detections 리스트로 변환 (score 0.3 이하 필터링 적용)
        detections = []
        for obj in msg.objects:
            if obj.score < 0.3:
                continue  # 🔥 Confidence 0.3 이하 detection은 버린다.

            detections.append({
                "position":    [obj.pos_x, obj.pos_y],
                "yaw":         obj.yaw,
                "size":        obj.size,
                "type":        obj.label,
                "reproj_bbox": obj.bbox_image,
            })

        # 3) GT 데이터에서 해당 token에 대한 GT 트랙 수 가져오기
        gt_tracks = self.gt_data.get(token, [])
        rospy.loginfo(f"[Tracker] GT Tracks for Token {token}: {len(gt_tracks)}")

        # 4) dt>0 일 때만 KF predict/update
        if dt > 0:
            self.tracker.predict(dt, self.ego_vel, self.ego_yaw_rate, self.ego_yaw)

            # Hungarian 매칭 수행 후 매칭된 트랙과 검출 객체를 순서대로 처리
            matches, unmatched_dets, unmatched_tracks, matched_tracks, matched_detections = \
                hungarian_iou_matching(self.tracker.tracks, detections)

            # 5) 매칭된 트랙을 업데이트
            for tr, det in zip(matched_tracks, matched_detections):
                tr.update(det, dt)
            
            # 6) 매칭되지 않은 트랙은 새로운 트랙으로 생성
            for di in unmatched_dets:
                self.tracker.tracks.append(KalmanTrackedObject(detections[di]))

            # 7) 죽은 트랙을 제거
            self.tracker.tracks = [
                t for t in self.tracker.tracks
                if t.missed_count <= t.max_missed
            ]
        else:
            rospy.logwarn(f"[Tracker] Skipping KF update for dt={dt:.3f}s")

        # 8) 트래킹된 객체 수 출력
        tracks = self.tracker.get_tracks()
        rospy.loginfo(f"[Tracker] Tracks Detected: {len(tracks)}")

        # 9) GT와 트래킹된 객체 수 비교
        rospy.loginfo(f"[Tracker] GT Tracks: {len(gt_tracks)}, Detected Tracks: {len(tracks)}")

        # 10) PfGMFATrackArray 생성 및 publish
        ta = PfGMFATrackArray()
        ta.header = msg.header
        for t in tracks:
            m = PfGMFATrack()
            m.pos_x         = t["x"]
            m.pos_y         = t["y"]
            m.yaw           = t["yaw"]
            dims            = list(t["size"])[:3]
            m.boundingbox   = dims + [0.0]*5
            m.confidence_ind= t["confidence"]
            m.id            = int(t["id"])
            m.obj_class     = t["type"]
            ta.tracks.append(m)

        self.tracking_pub.publish(ta)
        rospy.loginfo(f"[Tracker] Published {len(ta.tracks)} tracks")

        # 11) RViz 마커 퍼블리시 (tracking box + GT box)
        header = Header()
        header.frame_id = "map"
        header.stamp = rospy.Time.now()

        # 트래킹된 트랙 박스 퍼블리시
        marker_array = create_tracking_markers(tracks, header)
        self.vis_pub.publish(marker_array)

        # GT 박스 퍼블리시
        if gt_tracks:
            gt_marker_array = create_gt_markers(gt_tracks, header)
            self.vis_pub.publish(gt_marker_array)
            
if __name__ == '__main__':
    try:
        MCTrackTrackerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass