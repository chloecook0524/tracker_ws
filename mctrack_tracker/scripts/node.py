#!/usr/bin/env python3
import rospy
from std_msgs.msg import Header
from lidar_processing_msgs.msg import LidarPerceptionOutput, LidarObject
from visualization_msgs.msg import MarkerArray
from lidar_processing_msgs.msg import PfGMFATrackArray, PfGMFATrack
from mctrack_tracker_node import TrackerWorker, create_tracking_markers

def convert_msg_to_dict(obj: LidarObject):
    return {
        "position": [obj.pos_x, obj.pos_y],
        "yaw": obj.yaw,
        "size": list(obj.size),
        "type": obj.label,
        "velocity": [obj.vel_x, obj.vel_y],
        "reproj_bbox": list(obj.bbox_image),
        "confidence": obj.score
    }

class MCTNode:
    def __init__(self):
        rospy.init_node("tracker_node")
        cfg_path = rospy.get_param("~config_path", "/home/chloe/\ub2e4\uc6b4\ub85c\ub4dc/nuscenes.yaml")

        self.worker = TrackerWorker(cfg_path)
        self.token_seen_count = 0
        self.total_expected_tokens = rospy.get_param("/logger_expected_token_count", 0)

        # Subscribers
        rospy.Subscriber(
            "/lidar_detection",
            LidarPerceptionOutput,
            self.lidar_callback,
            queue_size=1, tcp_nodelay=True
        )

        # Publishers
        self.pub_tracks = rospy.Publisher("/tracking/objects", PfGMFATrackArray, queue_size=1)
        self.pub_vis = rospy.Publisher("/tracking/markers", MarkerArray, queue_size=1)

        rospy.Timer(rospy.Duration(0.05), self.on_timer)
        rospy.loginfo("tracker_node ready.")
        rospy.spin()

    def lidar_callback(self, msg: LidarPerceptionOutput):
        timestamp = msg.header.stamp.to_sec()
        detections = [convert_msg_to_dict(obj) for obj in msg.objects]
        cur_token = msg.header.frame_id

        self.worker.last_token = cur_token
        self.worker.append(timestamp, detections, token=cur_token)

        self.token_seen_count += 1
        if self.total_expected_tokens > 0:
            rospy.loginfo(f"[Tracker] ▶ Token Progress: {self.token_seen_count}/{self.total_expected_tokens}")
    
    def on_timer(self, event):
        tracks = self.worker.step()
        if not tracks:
            return

        now = rospy.Time.now()

        # Print GT vs Detected track counts
        if self.worker.last_token in self.worker.gt_data:
            gt_count = len(self.worker.gt_data[self.worker.last_token])
        else:
            gt_count = 0
        det_count = len(tracks)
        rospy.loginfo(f"[Tracker] GT Tracks: {gt_count}, Detected Tracks: {det_count}")

        # 1) Publish tracking result
        tracks_msg = PfGMFATrackArray()
        tracks_msg.header.stamp = now
        tracks_msg.header.frame_id = self.worker.last_token
        
        for t in tracks:
            tm = PfGMFATrack()
            tm.id       = int(t["id"])
            tm.pos_x    = float(t["x"])
            tm.pos_y    = float(t["y"])
            tm.yaw      = float(t["yaw"])

            # velocity, yaw_rate, accel placeholders
            tm.vel_x      = 0.0
            tm.yaw_rate   = 0.0
            tm.accel_x    = 0.0
            tm.yaw_accel  = 0.0

            size = t["size"]
            height = t.get("height", 1.5)
            bbox = list(size[:2]) + [height] + [0.0] * 5  # length 8
            tm.boundingbox = bbox

            tm.confidence_ind = float(t["confidence"])
            tm.obj_class = t["type"]
            tracks_msg.tracks.append(tm)

        self.pub_tracks.publish(tracks_msg)
        rospy.loginfo(f"[Tracker] Published {len(tracks_msg.tracks)} tracks (topic: /tracking/objects)")

        # 2) RViz tracking markers
        vis_msg = create_tracking_markers(tracks, tracks_msg.header)
        self.pub_vis.publish(vis_msg)


if __name__ == "__main__":
    MCTNode()