#!/usr/bin/env python3
import rospy
import json
import os
import signal
import tempfile
import shutil
from collections import defaultdict
from lidar_processing_msgs.msg import PfGMFATrackArray

# 평가에서 허용되는 클래스
ALLOWED_CLASSES = {
    1: "car", 2: "truck", 3: "bus", 4: "trailer",
    6: "pedestrian", 7: "motorcycle", 8: "bicycle"
}

class TrackingResultsLogger:
    def __init__(self):
        rospy.init_node("tracking_logger")

        raw_path = rospy.get_param("~output_path", "~/nuscenes_tracking_results.json")
        self.output_path = os.path.expanduser(raw_path)

        self.results = defaultdict(list)
        self.meta = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False
        }
        self.seen_tokens = set()

        self.sub = rospy.Subscriber("/tracking/objects", PfGMFATrackArray, self.callback)

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        rospy.loginfo(f"[Logger] ✅ Initialized. Saving to: {self.output_path}")

    def callback(self, msg: PfGMFATrackArray):
        sample_token = msg.header.frame_id
        if sample_token in self.seen_tokens:
            return
        self.seen_tokens.add(sample_token)

        # ✅ 빈 프레임도 기록 보장
        self.results[sample_token] = []

        for obj in msg.tracks:
            if obj.obj_class not in ALLOWED_CLASSES:
                continue
            tracking_name = ALLOWED_CLASSES[obj.obj_class]
            box = {
                "sample_token": sample_token,
                "translation": [obj.pos_x, obj.pos_y, 0.0],
                "size": obj.boundingbox[:3] if len(obj.boundingbox) >= 3 else [1.0, 1.0, 1.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "velocity": [obj.vel_x, 0.0],
                "tracking_id": str(obj.id),
                "tracking_name": tracking_name,
                "tracking_score": float(obj.confidence_ind)
            }
            self.results[sample_token].append(box)

    def spin(self):
        rate = rospy.Rate(1.0)
        try:
            while not rospy.is_shutdown():
                self.safe_save()
                rate.sleep()
        finally:
            self.safe_save()
            rospy.loginfo(f"[Logger] ✅ Final results saved to: {self.output_path}")

    def safe_save(self):
        if not self.results:
            return
        output = {
            "results": dict(self.results),
            "meta": self.meta
        }
        try:
            dir_name = os.path.dirname(self.output_path)
            with tempfile.NamedTemporaryFile('w', delete=False, dir=dir_name) as tmp_file:
                json.dump(output, tmp_file, indent=2)
                tmp_path = tmp_file.name
            shutil.move(tmp_path, self.output_path)
            rospy.loginfo(f"[Logger] Interim save complete: {self.output_path}")
        except Exception as e:
            rospy.logwarn(f"[Logger] ❌ Failed to save JSON: {e}")

    def signal_handler(self, signum, frame):
        rospy.logwarn(f"[Logger] Caught signal {signum}, exiting safely...")
        self.safe_save()
        rospy.signal_shutdown("Signal received")

if __name__ == "__main__":
    logger = TrackingResultsLogger()
    logger.spin() 