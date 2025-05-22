import rosbag
import matplotlib.pyplot as plt
from collections import defaultdict

bag_path = "/home/chloe/250513_bevfusion_detection/ioniq_ref_car_detection_result_2025-05-13-21-00-20.bag"
topic_name = "/detection_objects"


# 클래스별로 score 저장할 dict
class_scores = defaultdict(list)

# 라벨 매핑 (트래커에서 사용하던 것과 동일)
label_map = {
    "car": 1, "truck": 2, "bus": 3, "trailer": 4, "construction vehicle": 5,
    "pedestrian": 6, "motorcycle": 7, "bicycle": 8, "barrier": 9, "traffic cone": 10
}
id_to_name = {v: k for k, v in label_map.items()}

print("🔍 Reading bag...")
with rosbag.Bag(bag_path, 'r') as bag:
    count = 0
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        count += 1
        for obj in msg.objects:
            label = obj.label.strip().lower()
            label_id = label_map.get(label, -1)
            if label_id == -1:
                continue
            class_scores[label_id].append(obj.score)
        if count % 50 == 0:
            print(f"  {count} messages processed...")

print("✅ Parsing complete.")

# === 시각화 ===
plt.figure(figsize=(14, 10))
plot_count = len(class_scores)
cols = 3
rows = (plot_count + cols - 1) // cols

for idx, (label_id, scores) in enumerate(class_scores.items(), 1):
    plt.subplot(rows, cols, idx)
    plt.hist(scores, bins=30, range=(0.0, 0.05), color='skyblue', edgecolor='black')
    plt.title(f"{id_to_name.get(label_id, str(label_id)).title()} (n={len(scores)})")
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")

plt.tight_layout()
plt.suptitle("Per-Class Detection Confidence Distribution", fontsize=16, y=1.02)
plt.subplots_adjust(top=0.92)
plt.show()

