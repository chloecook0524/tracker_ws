import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import time

# 테스트용 이미지 로드 (예: 임의로 1280x720 컬러 이미지 생성)
img_np = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
final_dim = (256, 704)

# 1️⃣ OpenCV 방식 처리 함수
def process_with_opencv(img, final_dim):
    h, w = img.shape[:2]
    resize = final_dim[1] / w
    new_w = int(w * resize)
    new_h = int(h * resize)
    crop_h = int(new_h - final_dim[0])

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_cropped = img_resized[crop_h:crop_h + final_dim[0], 0:final_dim[1]]

    return img_cropped

# 2️⃣ PyTorch 방식 처리 함수
def process_with_torchvision(img_np, final_dim, use_cuda=False):
    img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0  # (C, H, W)
    if use_cuda:
        img = img.cuda()

    _, h, w = img.shape
    resize = final_dim[1] / w
    new_h = int(h * resize)
    new_w = int(w * resize)
    img_resized = TF.resize(img, [new_h, new_w], antialias=True)

    crop_h = int(new_h - final_dim[0])
    img_cropped = TF.crop(img_resized, crop_h, 0, final_dim[0], final_dim[1])

    if use_cuda:
        torch.cuda.synchronize()  # GPU 연산 완료 대기

    return img_cropped

# 실행 시간 측정 반복 횟수
n_iters = 100

# ✅ OpenCV 측정
start = time.time()
for _ in range(n_iters):
    out_cv2 = process_with_opencv(img_np, final_dim)
    out_cv2 = torch.from_numpy(out_cv2).cuda()
print(f"[OpenCV] 평균 처리 시간: {(time.time() - start) / n_iters:.6f}초")

# ✅ PyTorch (CPU)
start = time.time()
for _ in range(n_iters):
    out_torch_cpu = process_with_torchvision(img_np, final_dim, use_cuda=False)
    out_torch_cpu = out_torch_cpu.cuda()
print(f"[PyTorch CPU] 평균 처리 시간: {(time.time() - start) / n_iters:.6f}초")

# ✅ PyTorch (GPU, 가능할 때만)
if torch.cuda.is_available():
    start = time.time()
    for _ in range(n_iters):
        out_torch_gpu = process_with_torchvision(img_np, final_dim, use_cuda=True)
    print(f"[PyTorch GPU] 평균 처리 시간: {(time.time() - start) / n_iters:.6f}초")
else:
    print("[PyTorch GPU] CUDA 사용 불가 (GPU 없음)")


# import numpy as np
# import torch
# import time

# # PyTorch 필터링
# for i in range(5):
#     # 포인트 클라우드 100,000개 샘플 생성
#     pc_np1 = np.random.rand(100000, 3).astype(np.float32)
#     pc_np2 = np.random.rand(1000000, 3).astype(np.float32)
#     transform = np.eye(3, dtype=np.float32)

#     start = time.time()
#     pc_torch = torch.from_numpy(pc_np1).cuda()
#     pc_torch2 = torch.from_numpy(pc_np2).cuda()
#     transform_torch = torch.from_numpy(transform).cuda()
#     concat_pc_torch = torch.cat((pc_torch, pc_torch2@transform_torch), dim=0)
#     torch.cuda.synchronize()
#     print("Torch (GPU) time:", time.time() - start)

#     start = time.time()
#     pc_torch = torch.from_numpy(pc_np1)
#     pc_torch2 = torch.from_numpy(pc_np2)
#     transform_torch = torch.from_numpy(transform)
#     concat_pc_torch = torch.cat((pc_torch, pc_torch2@transform_torch), dim=0)
#     concat_pc_torch = concat_pc_torch.cuda()
#     print("Torch (cpU) time:", time.time() - start)

#     start = time.time()
#     concat_pc_np = np.concatenate((pc_np1, pc_np2@transform), axis=0)
#     concat_pc_np = torch.from_numpy(concat_pc_np).cuda()
#     print("NumPy time:", time.time() - start)
