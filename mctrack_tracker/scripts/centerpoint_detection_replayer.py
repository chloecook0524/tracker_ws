#!/usr/bin/env python3
import rospy
import json
import os
import numpy as np
from std_msgs.msg import Header, Float32
from lidar_processing_msgs.msg import LidarPerceptionOutput, LidarObject
from lidar_processing_msgs.msg import PfGMFATrackArray

# === Constants ===
VAL_JSON_PATH = "/home/chloe/SOTA/MCTrack/data/base_version/nuscenes/centerpoint/val.json"
GT_JSON_PATH = "/home/chloe/nuscenes_gt_valsplit.json"

CATEGORY_TO_LABEL = {
    "car": 1, "truck": 2, "bus": 3, "trailer": 4,
    "construction_vehicle": 5, "pedestrian": 6, "motorcycle": 7, "bicycle": 8,
    "traffic_cone": 9, "barrier": 10
}

# === Utility ===
def create_lidar_object(obj):
    lidar_obj = LidarObject()
    try:
        tracking_id = obj.get("tracking_id", "0")
        lidar_obj.id = abs(hash(tracking_id)) % (2 ** 16)
        label = CATEGORY_TO_LABEL.get(obj.get("category", "unknown"), 0)
        lidar_obj.label = label

        xyz = obj.get("global_xyz", [0.0, 0.0, 0.0])
        lidar_obj.pos_x = xyz[0]
        lidar_obj.pos_y = xyz[1]
        lidar_obj.pos_z = xyz[2]

        yaw = obj.get("global_yaw", 0.0)
        lidar_obj.yaw = yaw

        lwh = obj.get("lwh", [1.0, 1.0, 1.0])
        lidar_obj.size = [lwh[1], lwh[0], lwh[2]]  # width, length, height

        vel = obj.get("global_velocity", [0.0, 0.0])
        lidar_obj.vel_x = vel[0]
        lidar_obj.vel_y = vel[1]

        lidar_obj.score = obj.get("detection_score", 1.0)
        bbox_img = obj.get("bbox_image", {}).get("x1y1x2y2", [0.0]*4)
        lidar_obj.bbox_image = bbox_img if len(bbox_img) == 4 else [0.0]*4
    except Exception as e:
        rospy.logerr(f"Error creating lidar object: {e}")
        return None
    return lidar_obj

def wait_for_tracker_ready(max_attempts=5, timeout=5.0):
    import time
    attempt = 0
    while attempt < max_attempts and not rospy.is_shutdown():
        attempt += 1
        rospy.loginfo(f"⏳ Waiting for tracker to become active on /tracking/objects… (attempt {attempt}/{max_attempts})")
        try:
            rospy.wait_for_message("/tracking/objects", PfGMFATrackArray, timeout=timeout)
            rospy.loginfo("✅ Tracker is publishing. Safe to start replay.")
            return True
        except rospy.ROSException as e:
            rospy.logwarn(f"⚠️ Attempt {attempt} failed: {e}")
            time.sleep(1.0)
    rospy.logerr("❌ Tracker never became ready after retries; continuing replay anyway.")
    return False

# === Main ===
# Replacing the original main() loop and adding debugging logs
def main():
    try:
        rospy.init_node("val_replayer_node")
        rospy.loginfo("📦 [val_replayer_node] Starting with GT-based ego pose...")

        # 파일 존재 체크
        if not os.path.exists(VAL_JSON_PATH) or not os.path.exists(GT_JSON_PATH):
            rospy.logerr("❌ Required JSON files are missing.")
            return

        # JSON 로드
        with open(VAL_JSON_PATH, "r") as f:
            val_data = json.load(f)
        with open(GT_JSON_PATH, "r") as f:
            gt_data = json.load(f)

        ego_poses = gt_data.get("ego_poses", {})
        timestamps = gt_data.get("timestamps", {})

        # 일관성 체크
        val_tokens = set()
        for frames in val_data.values():
            for frame in frames:
                token = frame.get("cur_sample_token")
                if token:
                    val_tokens.add(token)
        missing_tokens = val_tokens - set(ego_poses.keys())
        if missing_tokens:
            rospy.logerr(f"❌ GT JSON is missing {len(missing_tokens)} tokens used in val.json.")
            rospy.signal_shutdown("Missing tokens in GT.")
            return
        else:
            rospy.loginfo(f"✅ All {len(val_tokens)} tokens are covered in GT.")

        # det_map 구성
        det_map = {}
        for token, ego in ego_poses.items():
            det_map[token] = {
                "bboxes": [],
                "ego_pose": ego,
                "timestamp": timestamps.get(token)
            }
        for frames in val_data.values():
            for frame in frames:
                token = frame.get("cur_sample_token")
                if token in det_map:
                    det_map[token]["bboxes"] = frame.get("bboxes", [])

        all_tokens = sorted(det_map.keys(), key=lambda t: det_map[t]["timestamp"] or 0)
        rospy.loginfo(f"🔍 Found {len(all_tokens)} tokens to publish.")

        # 퍼블리셔
        pub_det = rospy.Publisher("/lidar_detection", LidarPerceptionOutput, queue_size=1)
        pub_vel = rospy.Publisher("/ego_vel_x", Float32, queue_size=1)
        pub_yawrate = rospy.Publisher("/ego_yaw_rate", Float32, queue_size=1)
        pub_yaw = rospy.Publisher("/ego_yaw", Float32, queue_size=1)

        # 구독자 연결 대기
        # rospy.loginfo("⏳ Waiting for tracker to subscribe...")
        # while not rospy.is_shutdown() and (
        #     pub_det.get_num_connections() == 0 or
        #     pub_vel.get_num_connections() == 0 or
        #     pub_yawrate.get_num_connections() == 0 or
        #     pub_yaw.get_num_connections() == 0
        # ):
        #     rospy.sleep(0.1)
        rospy.loginfo("✅ All tracker subscribers detected.")
        wait_for_tracker_ready(max_attempts=1, timeout=3.0)

        # replay 루프
        rate = rospy.Rate(10.0)
        last_pose, last_ts = None, None

        for i, token in enumerate(all_tokens):
            if rospy.is_shutdown():
                rospy.loginfo(f"Shutting down at frame {i}.")
                break

            try:
                entry = det_map[token]
                bboxes = entry["bboxes"]
                ego_pose = entry["ego_pose"]
                timestamp = entry["timestamp"]

                # 메시지 생성
                msg = LidarPerceptionOutput()
                msg.header = Header()
                if timestamp is not None:
                    msg.header.stamp = rospy.Time.from_sec(timestamp)
                else:
                    msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = token

                if not isinstance(bboxes, list):
                    rospy.logerr(f"[cp_replayer] Expected list for bboxes, but got {type(bboxes)} at {token}.")
                    continue

                for j, box in enumerate(bboxes):
                    obj = create_lidar_object(box)
                    if obj:
                        msg.objects.append(obj)

                pub_det.publish(msg)

                # ego-pose → vel/yaw_rate
                if ego_pose and timestamp is not None:
                    cur_x, cur_y = ego_pose["translation"][:2]
                    q = ego_pose["rotation"]
                    yaw = np.arctan2(2*(q[0]*q[3] + q[1]*q[2]),
                                     1 - 2*(q[2]**2 + q[3]**2))
                    pub_yaw.publish(yaw)

                    if last_pose is not None:
                        dt = timestamp - last_ts
                        if dt > 0:
                            dx = cur_x - last_pose["x"]
                            dy = cur_y - last_pose["y"]
                            dyaw = yaw - last_pose["yaw"]
                            pub_vel.publish(np.hypot(dx, dy)/dt)
                            pub_yawrate.publish(dyaw/dt)

                    last_pose = {"x": cur_x, "y": cur_y, "yaw": yaw}
                    last_ts = timestamp

                rate.sleep()

            except Exception as e:
                rospy.logerr(f"[cp_replayer] Error at token={token}, frame={i}: {e}")
                rospy.logerr(traceback.format_exc())
                # 문제 생긴 프레임은 건너뛰고 계속 진행
                continue

        rospy.loginfo("🎉 Replay done — waiting 2s for any in-flight messages…")
        rospy.sleep(2.0)

    except rospy.ROSInterruptException:
        rospy.logwarn("⚠️ ROSInterruptException caught, shutting down cleanly.")
    except Exception as e:
        rospy.logerr(f"❌ Unhandled exception in main(): {e}")
        rospy.logerr(traceback.format_exc())
    finally:
        rospy.loginfo("🛑 val_replayer_node is exiting.")

if __name__ == "__main__":
    main()