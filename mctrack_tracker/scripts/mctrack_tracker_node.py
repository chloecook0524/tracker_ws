#!/usr/bin/env python3
import rospy
import numpy as np
import uuid
import json
from std_msgs.msg import Header, Float32
from lidar_processing_msgs.msg import LidarPerceptionOutput, LidarObject
from lidar_processing_msgs.msg import PfGMFATrack, PfGMFATrackArray
from scipy.optimize import linear_sum_assignment

# === Global Path to Baseversion Detection File ===
BASE_DET_JSON = "/home/chloe/SOTA/MCTrack/data/base_version/nuscenes/centerpoint/val.json"

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

# === Hungarian IoU Matching Function with predicted boxes ===
def hungarian_iou_matching(tracks, detections):
    if not tracks or not detections:
        return [], list(range(len(detections))), list(range(len(tracks)))

    cost_matrix = np.ones((len(tracks), len(detections)))
    for i, track in enumerate(tracks):
        pred_box = track.size[:2]
        for j, det in enumerate(detections):
            det_box = det["size"][:2]
            iou = iou_2d(pred_box, det_box)
            cost_matrix[i, j] = 1 - iou

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, unmatched_tracks, unmatched_dets = [], set(range(len(tracks))), set(range(len(detections)))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 0.9:
            matches.append((r, c))
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)

    return matches, list(unmatched_dets), list(unmatched_tracks)

# === TrackState Enum ===
class TrackState:
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3

# === Kalman Tracked Object Class ===
CLASS_CONFIG = {
    1: {"confirm_threshold": 1, "max_unmatch": 1},
    2: {"confirm_threshold": 1, "max_unmatch": 1},
    3: {"confirm_threshold": 1, "max_unmatch": 1},
    4: {"confirm_threshold": 1, "max_unmatch": 1},
    6: {"confirm_threshold": 1, "max_unmatch": 1},
    7: {"confirm_threshold": 1, "max_unmatch": 1},
    8: {"confirm_threshold": 1, "max_unmatch": 1},
}

# === KalmanTrackedObject (1/2) ===
class KalmanTrackedObject:
    def __init__(self, detection, obj_id=None):
        self.id = obj_id or uuid.uuid4().int % 65536
        self.label = detection['type']
        self.size = detection['size']
        self.x = detection['position'][0]
        self.y = detection['position'][1]
        self.yaw = detection['yaw']
        self.vx = 0.0
        self.yaw_rate = 0.0
        self.age = 0.0
        self.missed_count = 0
        self.last_update = rospy.Time.now().to_sec()
        self.hits = 1
        self.state = TrackState.TENTATIVE
        self.reproj_bbox = detection.get('reproj_bbox', None)
        cfg = CLASS_CONFIG.get(self.label, {"confirm_threshold": 3, "max_unmatch": 5})
        self.confirm_threshold = cfg["confirm_threshold"]
        self.max_unmatch = cfg["max_unmatch"]
        self.soft_deleted = False

    def predict(self, dt, ego_vel=0.0, ego_yaw_rate=0.0, ego_yaw=0.0):
        dx_ego = -ego_vel * dt * np.cos(ego_yaw)
        dy_ego = -ego_vel * dt * np.sin(ego_yaw)
        dyaw_ego = -ego_yaw_rate * dt
        self.x += self.vx * dt * np.cos(self.yaw) + dx_ego
        self.y += self.vx * dt * np.sin(self.yaw) + dy_ego
        self.yaw += self.yaw_rate * dt + dyaw_ego
        self.age += dt
        self.missed_count += 1


    def update(self, detection, dt):
        alpha = 0.5
        new_x = detection['position'][0]
        new_y = detection['position'][1]
        new_yaw = detection['yaw']
        new_size = detection['size']
        vx_est = (new_x - self.x) / dt if dt > 0 else 0.0
        yaw_rate_est = (new_yaw - self.yaw) / dt if dt > 0 else 0.0
        self.vx = alpha * vx_est + (1 - alpha) * self.vx
        self.yaw_rate = alpha * yaw_rate_est + (1 - alpha) * self.yaw_rate
        self.x = new_x
        self.y = new_y
        self.yaw = new_yaw
        self.size = [alpha * new_size[i] + (1 - alpha) * self.size[i] for i in range(3)]
        self.label = detection['type']
        self.reproj_bbox = detection.get('reproj_bbox', None)
        self.last_update = rospy.Time.now().to_sec()
        self.missed_count = 0
        self.hits += 1
        if self.hits >= self.confirm_threshold:
            self.state = TrackState.CONFIRMED

    def tracking_score(self):
        age_decay = np.exp(-0.1 * self.age)
        vel_consistency = np.exp(-abs(self.vx - 5) / 5.0)
        return max(0.1, min(1.0, (self.hits / (self.age + 1e-3)) * age_decay * vel_consistency))


# === KalmanMultiObjectTracker (predict only) ===
class KalmanMultiObjectTracker:
    def __init__(self, use_hungarian=False, use_reactivation=True, use_confidence_filtering=True, use_assistive_matching=True):
        self.tracks = []
        self.max_age = 1.2
        self.max_missed = 5

        self.use_hungarian = use_hungarian
        self.use_reactivation = use_reactivation
        self.use_confidence_filtering = use_confidence_filtering
        self.use_assistive_matching = use_assistive_matching

    def predict(self, dt, ego_vel=0.0, ego_yaw_rate=0.0, ego_yaw=0.0):
        for track in self.tracks:
            track.predict(dt, ego_vel, ego_yaw_rate, ego_yaw)

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
                iou = iou_2d(track.size[:2], det['size'][:2])
                yaw_sim = compute_yaw_similarity(track.yaw, det['yaw'])
                score = iou * yaw_sim

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
                iou = iou_2d(track.size[:2], det['size'][:2])
                yaw_sim = compute_yaw_similarity(track.yaw, det['yaw'])
                score = iou * yaw_sim
                if score > best_score and score > 0.3:
                    best_score = score
                    best_det = di
            if best_det >= 0:
                track.update(detections[best_det], dt)
                used_dets.add(best_det)


    def update(self, detections, dt):
        now = rospy.Time.now().to_sec()
        matches1, unmatched_dets1, unmatched_tracks1 = hungarian_iou_matching(self.tracks, detections)
        for ti, di in matches1:
            self.tracks[ti].update(detections[di], dt)

        matched_dets = set(di for _, di in matches1)
        matched_tracks = set(ti for ti, _ in matches1)

        rv_matches, unmatched_dets2, unmatched_tracks2 = image_plane_matching(
            [self.tracks[i] for i in unmatched_tracks1],
            [detections[i] for i in unmatched_dets1]
        )
        for local_ti, local_di in rv_matches:
            global_ti = list(unmatched_tracks1)[local_ti]
            global_di = list(unmatched_dets1)[local_di]
            self.tracks[global_ti].update(detections[global_di], dt)
            matched_dets.add(global_di)
            matched_tracks.add(global_ti)

        for di in range(len(detections)):
            if di not in matched_dets:
                self.tracks.append(KalmanTrackedObject(detections[di]))
        for ti in range(len(self.tracks)):
            if ti not in matched_tracks:
                self.tracks[ti].missed_count += 1

        for track in self.tracks:
            if now - track.last_update > self.max_age or track.missed_count > track.max_unmatch:
                track.soft_deleted = True

        if self.use_reactivation:
            self._reid_soft_deleted_tracks(detections, dt)

        if self.use_assistive_matching:
            self._fallback_match(unmatched_tracks1, unmatched_dets1, detections, dt)

    def get_tracks(self):
        results = []
        for t in self.tracks:
            if t.state != TrackState.CONFIRMED or t.soft_deleted:
                continue
            score = t.tracking_score()
            if self.use_confidence_filtering and score < 0.3:
                continue
            results.append({
                "id": t.id,
                "x": t.x,
                "y": t.y,
                "yaw": t.yaw,
                "size": t.size,
                "confidence": score,
                "type": t.label
            })
        return results

# === MCTrackTrackerNode (final integration) ===
class MCTrackTrackerNode:
    def __init__(self):
        rospy.init_node("mctrack_tracker_node", anonymous=True)
        self.tracker = KalmanMultiObjectTracker(
            use_hungarian=True,
            use_reactivation=True,
            use_confidence_filtering=False,
            use_assistive_matching=False
        )
        self.tracking_pub = rospy.Publisher("/tracking/objects", PfGMFATrackArray, queue_size=10)
        self.detection_sub = rospy.Subscriber("/lidar_detection", LidarPerceptionOutput, self.detection_callback, queue_size=1)
        self.vel_sub = rospy.Subscriber("/ego_vel_x", Float32, self.vel_callback, queue_size=1)
        self.yawrate_sub = rospy.Subscriber("/ego_yaw_rate", Float32, self.yawrate_callback, queue_size=1)
        self.yaw_sub = rospy.Subscriber("/ego_yaw", Float32, self.yaw_callback, queue_size=1)
        self.ego_vel = 0.0
        self.ego_yaw_rate = 0.0
        self.ego_yaw = 0.0
        self.last_time = rospy.Time.now()
        rospy.loginfo("MCTrackTrackerNode initialized and running.")

    def vel_callback(self, msg):
        self.ego_vel = msg.data

    def yawrate_callback(self, msg):
        self.ego_yaw_rate = msg.data

    def yaw_callback(self, msg):
        self.ego_yaw = msg.data

    def detection_callback(self, msg):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        detections = []
        for obj in msg.objects:
            det = {
                "position": [obj.pos_x, obj.pos_y],
                "yaw": obj.yaw,
                "size": obj.size,
                "type": obj.label,
                "reproj_bbox": obj.bbox_image,
            }
            detections.append(det)

        if dt <= 0:
            rospy.logwarn(f"[MCTrack] ⚠️ Non-positive dt={dt:.4f}. Skipping tracking update, but will publish empty or old tracks.")
        else:
            self.tracker.predict(dt, self.ego_vel, self.ego_yaw_rate, self.ego_yaw)
            self.tracker.update(detections, dt)

        tracks = self.tracker.get_tracks()
        track_array_msg = PfGMFATrackArray()
        track_array_msg.header = msg.header

        for track in tracks:
            trk_msg = PfGMFATrack()
            trk_msg.pos_x = track["x"]
            trk_msg.pos_y = track["y"]
            trk_msg.yaw = track["yaw"]
            trk_msg.boundingbox = list(track["size"]) + [0.0]*5
            trk_msg.confidence_ind = track["confidence"]
            trk_msg.id = int(track["id"])
            trk_msg.obj_class = track["type"]
            track_array_msg.tracks.append(trk_msg)

        self.tracking_pub.publish(track_array_msg)



    def convert_category_to_id(self, category):
        mapping = {
            "car": 1, "truck": 2, "bus": 3, "trailer": 4,
            "pedestrian": 6, "motorcycle": 7, "bicycle": 8
        }
        return mapping.get(category, 0)

if __name__ == '__main__':
    try:
        MCTrackTrackerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
