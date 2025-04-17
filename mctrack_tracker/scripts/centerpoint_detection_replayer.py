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
    # Convert [length, width, height] → [width, length, height]
    lidar_obj.size = [lwh[1], lwh[0], lwh[2]]

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

    pub_det = rospy.Publisher("/lidar_detection", LidarPerceptionOutput, queue_size=10)
    pub_vel = rospy.Publisher("/ego_vel_x", Float32, queue_size=1)
    pub_yawrate = rospy.Publisher("/ego_yaw_rate", Float32, queue_size=1)

    rospy.loginfo("📡 Waiting for subscribers to connect (min 3s)...")
    start_wait_time = rospy.Time.now().to_sec()
    while not rospy.is_shutdown():
        det_conn = pub_det.get_num_connections()
        vel_conn = pub_vel.get_num_connections()
        yaw_conn = pub_yawrate.get_num_connections()
        elapsed = rospy.Time.now().to_sec() - start_wait_time

        if det_conn > 0 and vel_conn > 0 and yaw_conn > 0 and elapsed > 3.0:
            rospy.loginfo(f"✅ Subscribers connected after {elapsed:.1f}s")
            break
        rospy.sleep(0.1)
    rospy.loginfo("✅ Subscribers connected. Beginning replay...")

    # === Flatten and sort all frames
    all_frames = []
    for scene_id, frames in val_data.items():
        all_frames.extend(frames)

    # 👉 정확한 sample_token 기준으로 정렬
    all_frames = sorted(all_frames, key=lambda f: f["cur_sample_token"])

    rate = rospy.Rate(10.0)
    last_pose = None
    last_ts = None
    start_time = rospy.Time.now()

    for i, frame in enumerate(all_frames):
        if rospy.is_shutdown():
            break

        token = frame.get("cur_sample_token", f"frame_{i}")
        bboxes = frame.get("bboxes", [])
        ego_pose = frame.get("ego_pose", {})
        timestamp = frame.get("timestamp", None)

        msg = LidarPerceptionOutput()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = token

        for box in bboxes:
            msg.objects.append(create_lidar_object(box))

        # === Ego velocity and yaw rate 계산
        if ego_pose and timestamp is not None:
            cur_x, cur_y = ego_pose.get("translation", [0.0, 0.0, 0.0])[:2]
            q = ego_pose.get("rotation", [1.0, 0.0, 0.0, 0.0])
            yaw = np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                             1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))

            if last_pose is not None and last_ts is not None:
                dt = (timestamp - last_ts)
                if dt > 0:
                    dx = cur_x - last_pose["x"]
                    dy = cur_y - last_pose["y"]
                    dyaw = yaw - last_pose["yaw"]
                    vel = np.hypot(dx, dy) / dt
                    yaw_rate = dyaw / dt
                    pub_vel.publish(vel)
                    pub_yawrate.publish(yaw_rate)

            last_pose = {"x": cur_x, "y": cur_y, "yaw": yaw}
            last_ts = timestamp

        pub_det.publish(msg)

        # 진행률 표시
        elapsed = (rospy.Time.now() - start_time).to_sec()
        progress = (i + 1) / len(all_frames)
        eta = (elapsed / progress) - elapsed
        rospy.loginfo(f"[{i+1}/{len(all_frames)}] Token: {token} | Progress: {progress*100:.1f}% | ETA: {eta:.1f}s")

        rate.sleep()

    rospy.loginfo("🎉 Finished publishing all baseversion detections.")

if __name__ == "__main__":
    main()
