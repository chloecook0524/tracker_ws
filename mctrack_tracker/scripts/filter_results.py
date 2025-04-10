#!/usr/bin/env python3
import argparse
import json
from tqdm import tqdm

# ✅ mctrack (NuScenes eval) 기준으로 인정되는 클래스만 남기기
VALID_TRACKING_NAMES = {
    "car", "truck", "bus", "trailer",
    "pedestrian", "motorcycle", "bicycle"
}

def filter_results(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    results = data["results"]
    meta = data["meta"]

    filtered_results = {}
    total_tokens = 0
    removed_objects = 0

    for token, objs in tqdm(results.items(), desc="Filtering"):
        total_tokens += 1
        valid_objs = []
        for obj in objs:
            if obj.get("tracking_name", "unknown") in VALID_TRACKING_NAMES:
                valid_objs.append(obj)
            else:
                removed_objects += 1
        # ✅ 프레임은 무조건 유지, 객체가 없으면 빈 리스트로
        filtered_results[token] = valid_objs

    print(f"\n✅ Total sample_tokens preserved: {total_tokens}")
    print(f"✅ Total objects removed due to unsupported tracking_name: {removed_objects}")

    output = {
        "results": filtered_results,
        "meta": meta
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✅ Filtered results saved to {output_path} (from {input_path})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    filter_results(args.input, args.output)
