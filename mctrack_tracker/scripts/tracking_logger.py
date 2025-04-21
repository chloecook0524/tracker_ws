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

        # 출력 파일 경로 설정
        raw_path = rospy.get_param("~output_path", "~/nuscenes_tracking_results.json")
        self.output_path = os.path.expanduser(raw_path)

        # 로깅할 데이터 구조 초기화
        self.results = defaultdict(list)
        self.meta = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False
        }
        self.seen_tokens = set()

        # 예상되는 모든 토큰을 저장하는 set
        self.expected_tokens = set()

        # param으로 넘겨받은 val.json 경로에서 토큰 추출
        val_json_path = rospy.get_param("~all_tokens_path", "/home/chloe/SOTA/MCTrack/data/base_version/nuscenes/centerpoint/val.json")
        self.load_expected_tokens(val_json_path)

        # 토픽 구독
        self.sub = rospy.Subscriber("/tracking/objects", PfGMFATrackArray, self.callback)

        # 신호 처리
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        rospy.loginfo(f"[Logger] ✅ Initialized. Saving to: {self.output_path}")

    def load_expected_tokens(self, val_json_path):
        """val.json에서 모든 토큰을 로드하여 expected_tokens에 저장"""
        try:
            with open(val_json_path, "r") as f:
                val_data = json.load(f)

            for scene_frames in val_data.values():
                for frame in scene_frames:
                    token = frame.get("cur_sample_token")
                    if token:
                        self.expected_tokens.add(token)

            rospy.loginfo(f"[Logger] 🧾 Loaded {len(self.expected_tokens)} expected sample tokens.")
        except Exception as e:
            rospy.logwarn(f"[Logger] ❌ Failed to load tokens from {val_json_path}: {e}")

    def callback(self, msg: PfGMFATrackArray):
        """트래킹 결과를 로깅하고, 누락된 토큰을 실시간으로 확인"""
        sample_token = msg.header.frame_id
        if sample_token in self.seen_tokens:
            return
        self.seen_tokens.add(sample_token)

        # 빈 프레임도 기록 보장
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

        # 실시간으로 누락된 토큰 체크
        if sample_token not in self.expected_tokens:
            rospy.logwarn(f"[Logger] ❗ Missing token: {sample_token}")

        if len(self.results) % 100 == 0:
            rospy.loginfo(f"[Logger] Logged {len(self.results)} / 6019 tokens so far.")

    def update_progress_file(self):
        """진행상황을 기록한 파일 업데이트"""
        try:
            progress_path = self.output_path.replace(".json", "_progress.txt")
            with open(progress_path, "w") as f:
                f.write(f"{len(self.results)} / 6019 tokens logged\n")
            if len(self.results) == 6019:
                rospy.loginfo("🎯 All 6019 tokens successfully logged!")
        except Exception as e:
            rospy.logwarn(f"[Logger] Failed to write progress file: {e}")

    def spin(self):
        """주기적으로 로깅을 저장"""
        rate = rospy.Rate(1.0)
        try:
            while not rospy.is_shutdown():
                self.safe_save()
                rate.sleep()
        finally:
            self.safe_save()
            rospy.loginfo(f"[Logger] ✅ Final results saved to: {self.output_path}")

    def safe_save(self):
        """결과를 안전하게 저장"""
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
            self.update_progress_file()
        except Exception as e:
            rospy.logwarn(f"[Logger] ❌ Failed to save JSON: {e}")

    def signal_handler(self, signum, frame):
        """시그널 처리 (예: Ctrl+C)"""
        rospy.logwarn(f"[Logger] Caught signal {signum}, exiting safely...")
        self.safe_save()
        rospy.signal_shutdown("Signal received")

if __name__ == "__main__":
    logger = TrackingResultsLogger()
    logger.spin()
