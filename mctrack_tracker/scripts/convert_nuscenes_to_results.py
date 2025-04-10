#!/usr/bin/env python3
import json
from collections import defaultdict
from tqdm import tqdm
import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

# === [ 경로 설정 ] ===
NUSC_ROOT = "/media/chloe/ec7602a4-b7fe-426f-bb59-0f9b8f98acb7"
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
    for scene in nusc.scene:
        if scene['name'] in val_scene_names:
            sample_token = scene['first_sample_token']
            while sample_token:
                val_sample_tokens.add(sample_token)
                sample = nusc.get('sample', sample_token)
                sample_token = sample['next'] if sample['next'] else None

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

        box = {
            'sample_token': ann['sample_token'],
            'translation': ann['translation'],
            'size': ann['size'],
            'rotation': ann['rotation'],
            'velocity': ann.get('velocity', [0.0, 0.0]),
            'tracking_id': ann['instance_token'],
            'tracking_name': tracking_name,
            'tracking_score': 1.0
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

if __name__ == "__main__":
    convert_val_gt_to_results_format()
