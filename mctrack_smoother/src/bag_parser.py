#!/usr/bin/env python3
import rosbag

def load_tracks_from_bag(bag_path, topic='/tracking/objects'):
    from lidar_processing_msgs.msg import PfGMFATrackArray
    import rospy
    import numpy as np

    bag = rosbag.Bag(bag_path)
    all_tracks = {}

    for topic, msg, t in bag.read_messages(topics=[topic]):
        timestamp = t.to_sec()
        for tr in msg.tracks:
            tid = tr.id
            if tid not in all_tracks:
                all_tracks[tid] = []
            all_tracks[tid].append({
                'time': timestamp,
                'x': tr.pos_x,
                'y': tr.pos_y,
                'yaw': tr.yaw,
                'size': tr.boundingbox[:3],
                'confidence': tr.confidence_ind
            })
    bag.close()
    return all_tracks
