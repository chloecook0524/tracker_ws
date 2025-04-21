#!/usr/bin/env python3
import rospy
import json
import os
import numpy as np
from std_msgs.msg import Header, Float32
from lidar_processing_msgs.msg import LidarPerceptionOutput, LidarObject
from lidar_processing_msgs.msg import PfGMFATrackArray  # for wait_for_message

# === Constants ===
VAL_JSON_PATH = "/home/chloe/SOTA/MCTrack/data/base_version/nuscenes/centerpoint/val.json"
GT_JSON_PATH = "/home/chloe/nuscenes_gt_valsplit.json"

CATEGORY_TO_LABEL = {
    "car": 1, "truck": 2, "bus": 3, "trailer": 4,
    "construction_vehicle": 5, "pedestrian": 6,
    "motorcycle": 7, "bicycle": 8,
    "traffic_cone": 9, "barrier": 10
}

# === Utility ===
def create_lidar_object(obj):
    lidar_obj = LidarObject()
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

    lidar_obj.score = obj.get("detection_score", 1.0)
    bbox_img = obj.get("bbox_image", {}).get("x1y1x2y2", [0.0]*4)
    lidar_obj.bbox_image = bbox_img if len(bbox_img) == 4 else [0.0]*4

    return lidar_obj

def wait_for_tracker_ready():
    rospy.loginfo("⏳ Waiting for tracker to become active on /tracking/objects...")
    try:
        rospy.wait_for_message("/tracking/objects", PfGMFATrackArray, timeout=10.0)
        rospy.loginfo("✅ Tracker is publishing. Safe to start replay.")
    except rospy.ROSException:
        rospy.logwarn("⚠️ Tracker did not publish in time. Proceeding anyway...")

# === Main ===
def main():
    rospy.init_node("val_replayer_node")
    rospy.loginfo("📦 [val_replayer_node] Starting with GT-based ego pose...")

    if not os.path.exists(VAL_JSON_PATH) or not os.path.exists(GT_JSON_PATH):
        rospy.logerr("❌ Required JSON files are missing.")
        return

    with open(VAL_JSON_PATH, "r") as f:
        val_data = json.load(f)
    with open(GT_JSON_PATH, "r") as f:
        gt_data = json.load(f)

    ego_poses = gt_data.get("ego_poses", {})
    timestamps = gt_data.get("timestamps", {})

    # === Consistency check ===
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

    # === Build detection map from GT + overlay detection ===
    det_map = {}
    for token, ego in ego_poses.items():
        det_map[token] = {
            "bboxes": [],
            "ego_pose": ego,
            "timestamp": timestamps.get(token, None)
        }

    for frames in val_data.values():
        for frame in frames:
            token = frame.get("cur_sample_token")
            if token in det_map:
                det_map[token]["bboxes"] = frame.get("bboxes", [])

    all_tokens = sorted(det_map.keys(), key=lambda t: det_map[t]["timestamp"] or 0)
    rospy.loginfo(f"🔍 Found {len(all_tokens)} tokens to publish.")

    # === Publishers ===
    pub_det = rospy.Publisher("/lidar_detection", LidarPerceptionOutput, queue_size=10)
    pub_vel = rospy.Publisher("/ego_vel_x", Float32, queue_size=1)
    pub_yawrate = rospy.Publisher("/ego_yaw_rate", Float32, queue_size=1)
    pub_yaw = rospy.Publisher("/ego_yaw", Float32, queue_size=1)

    # === Wait for subscribers ===
    rospy.loginfo("📡 Waiting for subscribers...")
    t0 = rospy.Time.now().to_sec()
    while not rospy.is_shutdown():
        if all(pub.get_num_connections() > 0 for pub in [pub_det, pub_vel, pub_yawrate, pub_yaw]) and rospy.Time.now().to_sec() - t0 > 3.0:
            rospy.loginfo("✅ Subscribers connected.")
            break
        rospy.sleep(0.1)

    # ✅ Wait until tracker starts publishing
    wait_for_tracker_ready()

    # === Replay loop ===
    rate = rospy.Rate(10.0)
    last_pose, last_ts = None, None
    start_time = rospy.Time.now()

    for i, token in enumerate(all_tokens):
        if rospy.is_shutdown():
            break

        entry = det_map[token]
        bboxes = entry["bboxes"]
        ego_pose = entry["ego_pose"]
        timestamp = entry["timestamp"]

        msg = LidarPerceptionOutput()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = token
        for box in bboxes:
            msg.objects.append(create_lidar_object(box))
        pub_det.publish(msg)

        # === Ego pose publish ===
        if ego_pose and timestamp is not None:
            cur_x, cur_y = ego_pose.get("translation", [0.0, 0.0, 0.0])[:2]
            q = ego_pose.get("rotation", [1.0, 0.0, 0.0, 0.0])
            yaw = np.arctan2(2.0 * (q[0]*q[3] + q[1]*q[2]),
                             1.0 - 2.0 * (q[2]**2 + q[3]**2))
            pub_yaw.publish(yaw)

            if last_pose and last_ts:
                dt = timestamp - last_ts
                if dt > 0:
                    dx = cur_x - last_pose["x"]
                    dy = cur_y - last_pose["y"]
                    dyaw = yaw - last_pose["yaw"]
                    pub_vel.publish(np.hypot(dx, dy) / dt)
                    pub_yawrate.publish(dyaw / dt)

            last_pose = {"x": cur_x, "y": cur_y, "yaw": yaw}
            last_ts = timestamp

        # === Progress log ===
        elapsed = (rospy.Time.now() - start_time).to_sec()
        progress = (i + 1) / len(all_tokens)
        eta = (elapsed / progress - elapsed) if progress > 0 else 0
        rospy.loginfo(f"[{i+1}/{len(all_tokens)}] {token} | {progress*100:.1f}% | ETA: {eta:.1f}s")

        rate.sleep()

    rospy.loginfo("🎉 Finished publishing all detections + ego poses.")

if __name__ == "__main__":
    main()