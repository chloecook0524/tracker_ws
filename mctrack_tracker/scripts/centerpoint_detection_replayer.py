#!/usr/bin/env python3
import rospy
import json
import os
import numpy as np
from std_msgs.msg import Header, Float32
from lidar_processing_msgs.msg import LidarPerceptionOutput, LidarObject

# === Constants ===
VAL_JSON_PATH = "/home/chloe/SOTA/MCTrack/data/base_version/nuscenes/centerpoint/val.json"

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
    lidar_obj.size = [lwh[1], lwh[0], lwh[2]]  # [width, length, height]

    lidar_obj.score = obj.get("detection_score", 1.0)
    bbox_img = obj.get("bbox_image", {}).get("x1y1x2y2", [0.0, 0.0, 0.0, 0.0])
    lidar_obj.bbox_image = bbox_img if len(bbox_img) == 4 else [0.0] * 4
    return lidar_obj


def main():
    rospy.init_node("val_replayer_node")
    rospy.loginfo("📦 [val_replayer_node] Starting up...")

    if not os.path.exists(VAL_JSON_PATH):
        rospy.logerr(f"❌ val.json not found at: {VAL_JSON_PATH}")
        return

    with open(VAL_JSON_PATH, "r") as f:
        val_data = json.load(f)

    # === Publishers ===
    pub_det = rospy.Publisher("/lidar_detection", LidarPerceptionOutput, queue_size=10)
    pub_vel = rospy.Publisher("/ego_vel_x", Float32, queue_size=10)
    pub_yawrate = rospy.Publisher("/ego_yaw_rate", Float32, queue_size=10)
    pub_yaw = rospy.Publisher("/ego_yaw", Float32, queue_size=10)

    rospy.loginfo("📡 Waiting for subscribers to connect (min 3s)...")
    start_wait = rospy.Time.now().to_sec()
    while not rospy.is_shutdown():
        if (pub_det.get_num_connections() > 0 and pub_vel.get_num_connections() > 0
                and pub_yawrate.get_num_connections() > 0 and pub_yaw.get_num_connections() > 0
                and rospy.Time.now().to_sec() - start_wait > 3.0):
            rospy.loginfo("✅ All subscribers connected.")
            break
        rospy.sleep(0.1)

    # === Build sample_token → frame mapping ===
    det_map = {}
    for frames in val_data.values():
        for frame in frames:
            token = frame.get("cur_sample_token")
            if token:
                det_map[token] = frame

    # === Sort tokens by timestamp ===
    all_tokens = sorted(det_map.keys(), key=lambda t: det_map[t].get("timestamp", 0))
    rospy.loginfo(f"🔍 Found {len(all_tokens)} unique sample_tokens to publish.")

    rate = rospy.Rate(10.0)
    last_pose = None
    last_ts = None
    start_time = rospy.Time.now()

    # === Publish loop ===
    for idx, token in enumerate(all_tokens):
        if rospy.is_shutdown():
            break

        frame = det_map[token]
        bboxes = frame.get("bboxes", [])
        ego = frame.get("ego_pose", {})
        ts = frame.get("timestamp")

        # Publish detection
        msg = LidarPerceptionOutput()
        msg.header = Header(stamp=rospy.Time.now(), frame_id=token)
        for box in bboxes:
            msg.objects.append(create_lidar_object(box))
        pub_det.publish(msg)

        # Publish ego motion: yaw, vel, yaw_rate
        if ego and ts is not None:
            cur_x, cur_y = ego.get("translation", [0.0, 0.0, 0.0])[:2]
            q = ego.get("rotation", [1.0, 0.0, 0.0, 0.0])
            cur_yaw = np.arctan2(2.0 * (q[0]*q[3] + q[1]*q[2]),
                                  1.0 - 2.0 * (q[2]**2 + q[3]**2))
            # Publish yaw
            pub_yaw.publish(cur_yaw)

            if last_pose is not None and last_ts is not None:
                dt = ts - last_ts
                if dt > 0:
                    dx = cur_x - last_pose["x"]
                    dy = cur_y - last_pose["y"]
                    dyaw = cur_yaw - last_pose["yaw"]
                    vel = np.hypot(dx, dy) / dt
                    yaw_rate = dyaw / dt
                    pub_vel.publish(vel)
                    pub_yawrate.publish(yaw_rate)

            last_pose = {"x": cur_x, "y": cur_y, "yaw": cur_yaw}
            last_ts = ts

        # Progress log
        elapsed = (rospy.Time.now() - start_time).to_sec()
        progress = (idx + 1) / len(all_tokens)
        eta = (elapsed / progress) - elapsed if progress > 0 else 0.0
        rospy.loginfo(f"[{idx+1}/{len(all_tokens)}] Token: {token} | {progress*100:.1f}% | ETA: {eta:.1f}s")

        rate.sleep()

    rospy.loginfo("🎉 Finished publishing all baseversion detections.")

if __name__ == "__main__":
    main()
