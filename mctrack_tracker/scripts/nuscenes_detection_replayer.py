#!/usr/bin/env python3
import rospy
import json
import os
import numpy as np
from std_msgs.msg import Header, Float32
from geometry_msgs.msg import Point
from lidar_processing_msgs.msg import LidarPerceptionOutput, LidarObject

# === Utility ===
def quaternion_to_yaw(q):
    qw, qx, qy, qz = q
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)

def create_lidar_object(obj, tracking_id):
    lidar_obj = LidarObject()
    lidar_obj.id = abs(hash(tracking_id)) % (2 ** 16)

    label_map = {
        "car": 1, "truck": 2, "bus": 3, "trailer": 4,
        "construction_vehicle": 5, "pedestrian": 6,
        "motorcycle": 7, "bicycle": 8, "traffic_cone": 9,
        "barrier": 10, "unknown": 0
    }
    label_str = obj.get("tracking_name", "unknown")
    lidar_obj.label = label_map.get(label_str, 0)

    lidar_obj.pos_x = obj["translation"][0]
    lidar_obj.pos_y = obj["translation"][1]
    lidar_obj.pos_z = obj["translation"][2]
    lidar_obj.size = obj["size"]
    lidar_obj.yaw = quaternion_to_yaw(obj["rotation"])
    lidar_obj.score = obj.get("tracking_score", 1.0)
    return lidar_obj

# === Main ===
def main():
    rospy.init_node("nuscenes_gt_publisher")
    rospy.loginfo(">>> Node initialized successfully")

    json_path = "/home/chloe/nuscenes_gt_valsplit.json"
    if not os.path.exists(json_path):
        rospy.logerr(f"GT JSON file does not exist at: {json_path}")
        return

    rospy.loginfo(f"Loading GT JSON from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    rospy.loginfo(">>> Successfully loaded JSON")

    pub_det = rospy.Publisher("/lidar_detection", LidarPerceptionOutput, queue_size=10)
    pub_vel = rospy.Publisher("/ego_vel_x", Float32, queue_size=1)
    pub_yawrate = rospy.Publisher("/ego_yaw_rate", Float32, queue_size=1)

    rospy.loginfo("⏳ Waiting for subscribers to connect...")
    while (pub_det.get_num_connections() == 0 or
           pub_vel.get_num_connections() == 0 or
           pub_yawrate.get_num_connections() == 0) and not rospy.is_shutdown():
        rospy.sleep(0.1)
    rospy.loginfo("✅ All subscribers connected. Starting replay.")

    rate = rospy.Rate(10.0)
    results = data["results"]
    ego_poses = data.get("ego_poses", {})
    timestamps = data["timestamps"]
    sample_tokens = sorted(results.keys(), key=lambda tok: timestamps.get(tok, 0))
    rospy.loginfo(f"Found {len(sample_tokens)} sample tokens.")

    last_pose = None
    last_ts = None
    start_time = rospy.Time.now()

    for i, token in enumerate(sample_tokens):
        if rospy.is_shutdown():
            break

        objects = results[token]
        msg = LidarPerceptionOutput()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = token

        # === Ego pose 기반 velocity, yaw rate 계산
        if token in ego_poses:
            cur_pose_raw = ego_poses[token]
            cur_x = cur_pose_raw["translation"][0]
            cur_y = cur_pose_raw["translation"][1]
            cur_yaw = quaternion_to_yaw(cur_pose_raw["rotation"])
            cur_ts = timestamps[token]

            if last_pose is not None:
                dt = cur_ts - last_ts
                if dt <= 0:
                    rospy.logwarn(f"[EGO DEBUG] ⚠️ Non-positive dt detected: {dt:.6f}. Skipping frame.")
                    continue
                dx = cur_x - last_pose["x"]
                dy = cur_y - last_pose["y"]
                dyaw = cur_yaw - last_pose["yaw"]

                vel_x = np.hypot(dx, dy) / dt if dt > 0 else 0.0
                yaw_rate = dyaw / dt if dt > 0 else 0.0

                # rospy.logwarn(f"[DEBUG EGO] dx={dx:.2f}, dy={dy:.2f}, dyaw={dyaw:.2f}, dt={dt:.4f}, vel={vel_x:.2f}, yaw_rate={yaw_rate:.2f}")
                pub_vel.publish(vel_x)
                pub_yawrate.publish(yaw_rate)

            last_pose = {"x": cur_x, "y": cur_y, "yaw": cur_yaw}
            last_ts = cur_ts

        for obj in objects:
            tracking_id = obj.get("tracking_id", "0")
            lidar_obj = create_lidar_object(obj, tracking_id)
            msg.objects.append(lidar_obj)

        pub_det.publish(msg)

        elapsed = (rospy.Time.now() - start_time).to_sec()
        progress = (i + 1) / len(sample_tokens)
        est_total = elapsed / progress
        eta = est_total - elapsed
        rospy.loginfo(f"[{i+1}/{len(sample_tokens)}] Published token: {token} | Progress: {progress*100:.2f}% | ETA: {eta:.1f}s")

        rate.sleep()

    rospy.loginfo("✅ Finished publishing all tokens. Shutting down...")

if __name__ == "__main__":
    main()
