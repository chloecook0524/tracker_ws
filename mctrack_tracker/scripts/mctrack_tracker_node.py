#!/usr/bin/env python3
import rospy
import numpy as np
import uuid
from std_msgs.msg import Header, Float32
from scipy.optimize import linear_sum_assignment
from lidar_processing_msgs.msg import LidarPerceptionOutput, LidarObject
from lidar_processing_msgs.msg import PfGMFATrack, PfGMFATrackArray

def compute_yaw_similarity(yaw1, yaw2):
    dyaw = abs(yaw1 - yaw2)
    return np.cos(dyaw)

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

    def predict(self, dt):
        self.x += self.vx * dt * np.cos(self.yaw)
        self.y += self.vx * dt * np.sin(self.yaw)
        self.yaw += self.yaw_rate * dt
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
        self.last_update = rospy.Time.now().to_sec()
        self.missed_count = 0
        self.hits += 1 

    def tracking_score(self):
        return max(0.1, min(1.0, self.hits / (self.age + 1e-6)))


class KalmanMultiObjectTracker:
    def __init__(self):
        self.tracks = []
        self.max_age = 2.0
        self.max_missed = 5

    def predict(self, dt):
        for track in self.tracks:
            track.predict(dt)

    def update(self, detections, dt):
        now = rospy.Time.now().to_sec()
        unmatched_dets = detections[:]

        for track in self.tracks:
            best_det = None
            best_score = 0.0
            for det in unmatched_dets:
                dist = np.linalg.norm([track.x - det['position'][0], track.y - det['position'][1]])
                iou = iou_2d(track.size, det['size'])
                yaw_sim = compute_yaw_similarity(track.yaw, det['yaw'])

                # print(f"[MATCHING DEBUG] Track {track.id} ↔ Det | dist: {dist:.2f}, iou: {iou:.3f}, yaw_sim: {yaw_sim:.3f}")

                if dist < 3.0 and iou > 0.01:
                    score = (1.0 / (dist + 1e-6)) + iou + yaw_sim
                    if score > best_score:
                        best_score = score
                        best_det = det

            if best_det:
                track.update(best_det, dt)
                unmatched_dets.remove(best_det)

        for det in unmatched_dets:
            self.tracks.append(KalmanTrackedObject(det))

        self.tracks = [t for t in self.tracks if (now - t.last_update < 2.0 and t.missed_count <= 5)]
    
    def compensate_ego_motion(self, dx, dy):
        for track in self.tracks:
            track.x += dx
            track.y += dy

    def get_tracks(self):
        result = []
        for t in self.tracks:
            # print(f"[TRACK DEBUG] ID: {t.id} | label: {t.label} | hits: {t.hits} | age: {t.age:.2f} | score: {t.tracking_score():.2f}")
            result.append({
                "id": t.id,
                "x": t.x,
                "y": t.y,
                "yaw": t.yaw,
                "size": t.size,
                "confidence": t.tracking_score(),
                "type": t.label
            })
        return result

class MCTrackTrackerNode:
    def __init__(self):
        rospy.init_node("mctrack_tracker_node", anonymous=True)

        self.tracker = KalmanMultiObjectTracker()
        self.tracking_pub = rospy.Publisher("/tracking/objects", PfGMFATrackArray, queue_size=10)

        self.detection_sub = rospy.Subscriber("/lidar_detection", LidarPerceptionOutput, self.detection_callback, queue_size=1)
        self.vel_sub = rospy.Subscriber("/ego_vel_x", Float32, self.vel_callback, queue_size=1)
        self.yawrate_sub = rospy.Subscriber("/ego_yaw_rate", Float32, self.yawrate_callback, queue_size=1)

        self.ego_vel = 0.0
        self.ego_yaw_rate = 0.0
        self.last_time = rospy.Time.now()

        rospy.loginfo("MCTrackTrackerNode initialized and running.")

    def vel_callback(self, msg):
        self.ego_vel = msg.data

    def yawrate_callback(self, msg):
        self.ego_yaw_rate = msg.data

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
            }
            detections.append(det)

        dx = -self.ego_vel * dt * np.cos(self.ego_yaw_rate * dt)
        dy = -self.ego_vel * dt * np.sin(self.ego_yaw_rate * dt)
        # self.tracker.compensate_ego_motion(dx, dy)
        # rospy.logwarn(f"[EGO MOTION] dx: {dx:.2f}, dy: {dy:.2f}, vel: {self.ego_vel:.2f}, yaw_rate: {self.ego_yaw_rate:.2f}, dt: {dt:.2f}")


        self.tracker.predict(dt)
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

if __name__ == '__main__':
    try:
        MCTrackTrackerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass