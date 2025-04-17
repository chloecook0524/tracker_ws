#!/usr/bin/env python3
import json
from collections import defaultdict
from tqdm import tqdm
import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
import numpy as np
from pyquaternion import Quaternion

# === [ 경로 설정 ] ===
NUSC_ROOT = "/media/chloe/ec7602a4-b7fe-426f-bb59-0f9b8f98acb72"
VERSION = "v1.0-trainval"
OUTPUT_JSON_PATH = "/home/chloe/nuscenes_gt_valsplit.json"

def convert_category_name_to_tracking_name(category_name):
    if category_name.startswith("vehicle.car"):
        return "car"
    elif category_name.startswith("vehicle.truck"):
        return "truck"
    elif category_name.startswith("vehicle.bus"):
        return "bus"
    elif category_name.startswith("vehicle.trailer"):
        return "trailer"
    elif category_name.startswith("vehicle.construction"):
        return "construction_vehicle"
    elif category_name.startswith("human.pedestrian"):
        return "pedestrian"
    elif category_name.startswith("vehicle.motorcycle"):
        return "motorcycle"
    elif category_name.startswith("vehicle.bicycle"):
        return "bicycle"
    elif category_name.startswith("movable_object.trafficcone"):
        return "traffic_cone"
    elif category_name.startswith("movable_object.barrier"):
        return "barrier"
    return None

def convert_val_gt_to_results_format():
    print(f"📦 Loading NuScenes from {NUSC_ROOT} ...")
    nusc = NuScenes(version=VERSION, dataroot=NUSC_ROOT)

    val_scene_names = create_splits_scenes()['val']

    val_sample_tokens = set()
    val_scene_names = create_splits_scenes()['val']
    for scene in nusc.scene:
        if scene['name'] in val_scene_names:
            sample_token = scene['first_sample_token']
            while sample_token != '':
                sample = nusc.get('sample', sample_token)
                val_sample_tokens.add(sample_token)
                sample_token = sample['next']

    results = defaultdict(list)
    ego_poses = dict()
    count_valid = 0
    skipped = 0

    print(f"🔁 Filtering annotations from val split ({len(val_sample_tokens)} samples)...")
    for ann in tqdm(nusc.sample_annotation, desc="Converting annotations"):
        if ann['sample_token'] not in val_sample_tokens:
            continue

        category_name = ann.get("category_name", "")
        tracking_name = convert_category_name_to_tracking_name(category_name)
        if tracking_name is None:
            skipped += 1
            continue

        # === Reproject to image plane (CAM_FRONT)
        sample = nusc.get('sample', ann['sample_token'])
        cam_token = sample['data'].get('CAM_FRONT', None)
        bbox_reproj = None
        if cam_token:
            cam_sd = nusc.get('sample_data', cam_token)
            cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
            cam_intrinsic = np.array(cam_cs['camera_intrinsic'])

            # Box in global -> sensor frame
            quat = Quaternion(ann['rotation'])   # 회전 값을 Quaternion 객체로 변환
            box3d = Box(ann['translation'], ann['size'], quat)
            box3d.translate(-np.array(cam_cs['translation']))
            box3d.rotate(Quaternion(cam_cs['rotation']).inverse)

            corners = view_points(box3d.corners(), cam_intrinsic, normalize=True)
            x1, y1 = np.min(corners[:2], axis=1)
            x2, y2 = np.max(corners[:2], axis=1)
            bbox_reproj = [float(x1), float(y1), float(x2), float(y2)]

        # === Output entry
        box = {
            'sample_token': ann['sample_token'],
            'translation': ann['translation'],
            'size': ann['size'],
            'rotation': ann['rotation'],
            'velocity': ann.get('velocity', [0.0, 0.0]),
            'tracking_id': ann['instance_token'],
            'tracking_name': tracking_name,
            'tracking_score': 1.0,
            'bbox_image': {
                'x1y1x2y2': bbox_reproj
            }
        }

        results[ann['sample_token']].append(box)
        count_valid += 1

    timestamps = dict()
    print("📍 Extracting ego_pose per sample token...")
    for tok in tqdm(val_sample_tokens, desc="Ego pose"):
        sample = nusc.get('sample', tok)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])

        ego_poses[tok] = {
            "translation": ego_pose['translation'],
            "rotation": ego_pose['rotation']
        }
        timestamps[tok] = lidar_data['timestamp'] / 1e6 

    for tok in val_sample_tokens:
        if tok not in results:
            results[tok] = []

    output = {
        'results': dict(results),
        'ego_poses': ego_poses,
        'timestamps': timestamps,
        'meta': {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False
        }
    }

    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Extracted {count_valid} valid annotations")
    print(f"⚠️ Skipped: {skipped}")
    print(f"✅ Saved final output to: {OUTPUT_JSON_PATH}")
    print(f"Checking first few val sample tokens: {list(val_sample_tokens)[:5]}")

if __name__ == "__main__":
    convert_val_gt_to_results_format()
