#!/usr/bin/env python3
import rospy
import json
import os
import signal
import tempfile
import shutil
import threading
from collections import defaultdict
from lidar_processing_msgs.msg import PfGMFATrackArray

# 평가에서 허용되는 클래스 매핑
ALLOWED_CLASSES = {
    1: "car",
    2: "truck",
    3: "bus",
    4: "trailer",
    6: "pedestrian",
    7: "motorcycle",
    8: "bicycle"
}

class TrackingResultsLogger:
    def __init__(self):
        # 노드 초기화
        rospy.init_node("tracking_logger")
        # 종료 시 최종 저장
        rospy.on_shutdown(self.safe_save)

        # 로거가 준비되었음을 나타내는 플래그 세팅
        rospy.set_param("/logger_ready", True)
        rospy.loginfo("[Logger] ▶ /logger_ready = true")

        # 출력 파일 경로
        raw_path = rospy.get_param("~output_path", "~/nuscenes_tracking_results.json")
        self.output_path = os.path.expanduser(raw_path)

        # 내부 데이터 구조
        self.results = defaultdict(list)
        self.meta = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False
        }
        self.seen_tokens = set()
        self.lock = threading.Lock()

        # 기대되는 토큰 목록
        self.expected_tokens = set()
        val_json = rospy.get_param(
            "~all_tokens_path",
            "/home/chloe/SOTA/MCTrack/data/base_version/nuscenes/centerpoint/val.json"
        )
        self._load_expected_tokens(val_json)

        # 토픽 구독 (버퍼 크기를 토큰 수만큼)
        self.sub = rospy.Subscriber(
            "/tracking/objects",
            PfGMFATrackArray,
            self.callback,
            queue_size=len(self.expected_tokens)
        )

        # SIGINT/SIGTERM 처리
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        rospy.loginfo(f"[Logger] Initialized. Expecting {len(self.expected_tokens)} tokens.")
        # 즉시 진행상황 파일 갱신
        self.update_progress_file()

        # 1초마다 안전 저장
        rospy.Timer(rospy.Duration(1.0), lambda evt: self.safe_save())
        
    def _load_expected_tokens(self, path):
        try:
            with open(path, "r") as f:
                val_data = json.load(f)
            for scene in val_data.values():
                for frame in scene:
                    tk = frame.get("cur_sample_token")
                    if tk:
                        self.expected_tokens.add(tk)
            rospy.loginfo(f"[Logger] Loaded {len(self.expected_tokens)} expected tokens.")
        except Exception as e:
            rospy.logwarn(f"[Logger] Failed to load tokens from {path}: {e}")

    def callback(self, msg: PfGMFATrackArray):
        token = msg.header.frame_id
        with self.lock:
            if token in self.seen_tokens:
                return
            self.seen_tokens.add(token)
            self.results[token] = []
            for obj in msg.tracks:
                if obj.obj_class not in ALLOWED_CLASSES:
                    continue
                box = {
                    "sample_token": token,
                    "translation": [obj.pos_x, obj.pos_y, 0.0],
                    "size": obj.boundingbox[:3] if len(obj.boundingbox) >= 3 else [1.0,1.0,1.0],
                    "rotation": [0.0,0.0,0.0,1.0],
                    "velocity": [getattr(obj, 'vel_x', 0.0), 0.0],
                    "tracking_id": str(obj.id),
                    "tracking_name": ALLOWED_CLASSES[obj.obj_class],
                    "tracking_score": float(obj.confidence_ind)
                }
                self.results[token].append(box)

        # 진행상황 업데이트
        self.update_progress_file()
        if token not in self.expected_tokens:
            rospy.logwarn(f"[Logger] Unexpected token seen: {token}")

    def update_progress_file(self):
        prog_path = self.output_path.replace(".json", "_progress.txt")
        line = f"{len(self.results)} / {len(self.expected_tokens)} tokens logged\n"
        try:
            with open(prog_path, "w") as f:
                f.write(line)
        except Exception as e:
            rospy.logwarn(f"[Logger] Could not write progress file: {e}")

    def safe_save(self):
        with self.lock:
            if not self.results:
                return
            data = {
                "results": dict(self.results),
                "meta": self.meta
            }
        try:
            tmp_dir = os.path.dirname(self.output_path)
            with tempfile.NamedTemporaryFile("w", delete=False, dir=tmp_dir) as tf:
                json.dump(data, tf, indent=2)
                tmp_name = tf.name
            shutil.move(tmp_name, self.output_path)
            rospy.logdebug(f"[Logger] Saved JSON to {self.output_path}")
        except Exception as e:
            rospy.logwarn(f"[Logger] Failed to save JSON: {e}")

    def _signal_handler(self, signum, frame):
        rospy.logwarn(f"[Logger] Caught signal {signum}, flushing...")
        deadline = rospy.Time.now() + rospy.Duration(5.0)
        while len(self.results) < len(self.expected_tokens) and rospy.Time.now() < deadline:
            rospy.sleep(0.05)
        self.safe_save()
        rospy.signal_shutdown("shutdown after flush")

if __name__ == "__main__":
    TrackingResultsLogger()
    rospy.spin()
