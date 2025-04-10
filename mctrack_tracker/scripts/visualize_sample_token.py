#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt

# === [파일 경로 설정] ===
gt_json_path = "/home/chloe/nuscenes_gt_valsplit.json"
pred_json_path = "/home/chloe/nuscenes_tracking_results.json"
sample_token = "000681a060c04755a1537cf83b53ba57"

# === [JSON 로딩] ===
with open(gt_json_path) as f:
    gt_data = json.load(f)
with open(pred_json_path) as f:
    pred_data = json.load(f)

gt_boxes = gt_data["results"].get(sample_token, [])
pred_boxes = pred_data["results"].get(sample_token, [])

print(f"[INFO] GT boxes: {len(gt_boxes)}, Prediction boxes: {len(pred_boxes)}")

print("\n[GT box translations]:")
for b in gt_boxes:
    print(b["translation"])

print("\n[Prediction box translations]:")
for b in pred_boxes:
    print(b["translation"])

# === [시각화] ===
plt.figure()
ax = plt.gca()
ax.set_aspect("equal")

# GT (green)
for box in gt_boxes:
    x, y = box["translation"][:2]
    plt.plot(x, y, 'go', label="GT" if 'GT' not in ax.get_legend_handles_labels()[1] else "")

# Prediction (red)
for box in pred_boxes:
    x, y = box["translation"][:2]
    plt.plot(x, y, 'ro', label="Prediction" if 'Prediction' not in ax.get_legend_handles_labels()[1] else "")

# 자동 축 설정
all_x = [b["translation"][0] for b in gt_boxes + pred_boxes]
all_y = [b["translation"][1] for b in gt_boxes + pred_boxes]
if all_x and all_y:
    margin = 10
    plt.xlim(min(all_x) - margin, max(all_x) + margin)
    plt.ylim(min(all_y) - margin, max(all_y) + margin)

plt.title(f"sample_token: {sample_token}")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()
