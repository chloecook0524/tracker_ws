#!/usr/bin/env python
"""
ROS BEVFusion Inference Node (restructured)
- LiDAR 콜백: 포인트·Extrinsics 버퍼 갱신, 최신 stamp(work_q maxsize=1)에 push
- 워커 스레드: stamp 기준으로 이미지·TF 재조회 후 mixed‑precision 추론
- TimeBuffer: bisect O(logN) 검색
- 버퍼·모델 입력 모두 float16, torch.cuda.amp 활용
"""

import bisect, queue, threading, time
from collections import deque

import cv2
import numpy as np
import rospy
import torch
import torch.cuda.amp as amp
import torchvision.transforms.functional as TF
from pyquaternion import Quaternion
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Header
from vdcl_fusion_perception.msg import DetectionResult
import matplotlib.pyplot as plt

from mmengine import Config
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes

# -----------------------------------------------------------------------------
# 1. TimeBuffer ----------------------------------------------------------------
# -----------------------------------------------------------------------------
class TimeBuffer:
    def __init__(self, maxlen=100):
        self.stamps: list[float] = []
        self.data:   list       = []
        self.maxlen = maxlen

    def add(self, stamp: float, data):
        idx = bisect.bisect_left(self.stamps, stamp)
        self.stamps.insert(idx, stamp)
        self.data.insert(idx, data)
        if len(self.stamps) > self.maxlen:
            self.stamps.pop(0)
            self.data.pop(0)

    def get_closest(self, target: float, max_dt=0.2):
        if not self.stamps:
            return None, None
        idx = bisect.bisect_left(self.stamps, target)
        cand = []
        if idx < len(self.stamps): cand.append(idx)
        if idx > 0: cand.append(idx-1)
        best = min(cand, key=lambda i: abs(self.stamps[i]-target)) if cand else None
        if best is not None and abs(self.stamps[best]-target) <= max_dt:
            return self.data[best], self.stamps[best]
        return None, None

# -----------------------------------------------------------------------------
# 2. Sensor‑specific helpers ----------------------------------------------------
# -----------------------------------------------------------------------------
CAM_KEYS = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]

class Meta:
    def __init__(self, cam_keys):
        self.cam_keys = cam_keys
        self.cam_intrinsics = {k: np.eye(4, dtype=np.float32) for k in cam_keys}
        self.cam_extrinsics = {k: np.eye(4, dtype=np.float32) for k in cam_keys}
        self.lidar_extrinsics = np.eye(4, dtype=np.float32)
        self.all_img_aug_matrix = np.repeat([np.eye(4, dtype=np.float32)], 6, axis=0)
        self.sample_idx = 0

    @staticmethod
    def tfmat(rotation, translation):
        T = np.eye(4, dtype=np.float32)
        T[:3,:3] = Quaternion(rotation).rotation_matrix
        T[:3,3]  = translation
        return T

    def add_cam_intr(self, cam, K):
        K = np.array(K).reshape(3,3)
        self.cam_intrinsics[cam] = np.block([[K, np.zeros((3,1))],[np.zeros((1,4))]])

    def as_dict(self):
        stacks = {k:[] for k in ['lidar2cam','cam2lidar','cam2img','lidar2img']}
        for cam in self.cam_keys:
            l2c = np.linalg.inv(self.cam_extrinsics[cam]) @ self.lidar_extrinsics
            c2l = np.linalg.inv(l2c)
            c2i = self.cam_intrinsics[cam]
            l2i = c2i @ l2c
            stacks['lidar2cam'].append(l2c)
            stacks['cam2lidar'].append(c2l)
            stacks['cam2img'].append(c2i)
            stacks['lidar2img'].append(l2i)
        meta = {'sample_idx':self.sample_idx,'num_pts_feats':5,'box_type_3d':LiDARInstance3DBoxes,'img_aug_matrix':self.all_img_aug_matrix}
        meta.update({k:np.stack(v) for k,v in stacks.items()})
        self.sample_idx = (self.sample_idx+1)%10000
        return meta

class LiDARSweeps:
    def __init__(self, num_sweeps=10):
        self.num_sweeps=num_sweeps
        self.points, self.ts = deque(maxlen=num_sweeps), deque(maxlen=num_sweeps)
        self.e2g, self.l2e = deque(maxlen=num_sweeps), deque(maxlen=num_sweeps)

    def add(self, pts, l2e, e2g, stamp):
        # self.points.append(pts.half())
        self.points.append(pts.float())
        # self.l2e.append(None if l2e is None else torch.from_numpy(l2e).cuda().half())
        self.l2e.append(None if l2e is None else torch.from_numpy(l2e).cuda().float())
        # self.e2g.append(None if e2g is None else torch.from_numpy(e2g).cuda().half())
        self.e2g.append(None if e2g is None else torch.from_numpy(e2g).cuda().float())
        self.ts.append(stamp)

    def build(self):
        points = list(self.points)
        ts = list(self.ts)
        l2e = list(self.l2e)
        e2g = list(self.e2g)

        base = points[-1]
        mask = ~((base[:, 0] >= -2) & (base[:, 0] <= 3) & (base[:, 1] >= -1.2) & (base[:, 1] <= 1.2))
        base = base[mask]

        # zeros = torch.zeros((base.shape[0], 1), device='cuda', dtype=torch.float16)
        zeros = torch.zeros((base.shape[0], 1), device='cuda', dtype=torch.float32)
        combined = [torch.cat([base, zeros], 1)]

        if self.num_sweeps == 1:
            return [combined[0]]
        e2g_cur, l2e_cur = e2g[-1], l2e[-1]
        if e2g_cur is None or l2e_cur is None:
            return [combined[0]]

        t_cur = ts[-1]

        for i in range(len(points) - 1):
            if t_cur - ts[i] > 0.6 or e2g[i] is None or l2e[i] is None:
                continue
            xyz, inten = points[i][:, :3], points[i][:, 3:4]
            # ones = torch.ones((xyz.shape[0], 1), device='cuda', dtype=torch.float16)
            ones = torch.ones((xyz.shape[0], 1), device='cuda', dtype=torch.float32)
            pts_h = torch.cat([xyz, ones], 1)
            pts_g = (pts_h @ l2e[i].T) @ e2g[i].T
            # g2e_cur = torch.inverse(e2g_cur.float()).half()
            g2e_cur = torch.inverse(e2g_cur.float())
            # e2l_cur = torch.inverse(l2e_cur.float()).half()
            e2l_cur = torch.inverse(l2e_cur.float())
            pts_c = (pts_g @ g2e_cur.T) @ e2l_cur.T
            # dt = torch.full((xyz.shape[0], 1), t_cur - ts[i], device='cuda', dtype=torch.float16)
            dt = torch.full((xyz.shape[0], 1), t_cur - ts[i], device='cuda', dtype=torch.float32)
            combined.append(torch.cat([pts_c[:, :3], inten, dt], 1))

        return [torch.cat(combined, 0).half()]

class CamImages:
    def __init__(self, cam_keys, shape=(256,704)):
        self.shape=shape
        self.buffers={k:TimeBuffer(200) for k in cam_keys}
        self.cam_keys=cam_keys

    def add(self, tensor, cam, ts): 
        self.buffers[cam].add(ts,tensor)

    def _prep(self,img):
        _,h,w=img.shape
        H,W=self.shape
        r=W/w
        img=TF.resize(img,[int(h*r),int(w*r)],antialias=True)
        crop=int(img.shape[1]-H)
        img=TF.crop(img,crop,0,H,W)
        aug=np.eye(4,dtype=np.float32)
        aug[:2,:2]*=r
        aug[:2,3]=[0,-crop]
        return img.half(),aug
        # return img.float(),aug
    
    def fetch(self, target,max_dt=0.2):
        imgs,augs=[],[]
        for k in self.cam_keys:
            im,_=self.buffers[k].get_closest(target,max_dt)
            # im = None
            if im is None:
                imgs.append(torch.zeros((3,*self.shape),device='cuda',dtype=torch.float16))
                # imgs.append(torch.zeros((3,*self.shape),device='cuda',dtype=torch.float32))
                augs.append(np.eye(4,dtype=np.float32))
            else:
                t,a=self._prep(im)
                imgs.append(t)
                augs.append(a)
        return torch.stack(imgs).unsqueeze(0), np.stack(augs)

# -----------------------------------------------------------------------------
# 3. Inference Node ------------------------------------------------------------
# -----------------------------------------------------------------------------
class InferenceNode:
    def __init__(self):
        rospy.init_node('inference_node')
        self.model=self._load_model().eval().half()
        # self.model=self._load_model().eval()
        self.meta=Meta(CAM_KEYS)

        self.ego_tf=TimeBuffer()
        self.cam_tf={k:TimeBuffer() for k in CAM_KEYS}
        self.lidar_tf=TimeBuffer()
        self.cams=CamImages(CAM_KEYS)
        self.lidar=LiDARSweeps(num_sweeps=10)

        self.work_q=queue.Queue(maxsize=1)
        threading.Thread(target=self._worker,daemon=True).start()

        for cam in CAM_KEYS:
            rospy.Subscriber(cam.lower(), Image, self._img_cb, callback_args=cam, queue_size=1)
            # rospy.Subscriber(f"{cam.lower()}/image_raw", Image, self._img_cb, callback_args=cam, queue_size=1)
            rospy.Subscriber(f"{cam.lower()}_info", CameraInfo, self._cam_info_cb, callback_args=cam, queue_size=1)
        rospy.Subscriber('tf', TFMessage, self._tf_cb, queue_size=1)
        rospy.Subscriber('lidar_points', PointCloud2, self._lidar_cb, queue_size=1)
        rospy.Subscriber('velodyne_pointcloud', PointCloud2, self._lidar_cb, queue_size=1)

        self.det_pub=rospy.Publisher('detection_results', DetectionResult, queue_size=1)
        self.lidar_pub=rospy.Publisher('detection_lidar_points', PointCloud2, queue_size=1)

        self.total_time = 0
        self.count = 0

    # ---------------- model ------------------
    def _load_model(self):
        cfg = Config.fromfile('/home/sgsp/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0100_depth_with_lidar_depth.py')
        model = MODELS.build(cfg.model)
        load_checkpoint(model,
                        '/home/sgsp/mmdetection3d/work_dirs/bevfusion_lidar-cam_voxel0100_depth_with_lidar_depth_400q/epoch_6.pth',
                        map_location='cpu')
        # cfg=Config.fromfile('/home/sgsp/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0100_depth_from_scratch.py')
        # model=MODELS.build(cfg.model)
        # load_checkpoint(model,'/home/sgsp/mmdetection3d/work_dirs/bevfusion_lidar-cam_voxel0100_depth_from_scratch/epoch_6.pth',map_location='cpu')
        return model.cuda()

    # ---------------- Callbacks -------------
    def _img_cb(self,msg,cam):
        img=np.frombuffer(msg.data,np.uint8).reshape((msg.height,msg.width,3))[:,:,::-1].copy()
        t=torch.from_numpy(img).cuda().permute(2,0,1)
        self.cams.add(t,cam,msg.header.stamp.to_sec())

    def _cam_info_cb(self,msg,cam):
        if np.all(self.meta.cam_intrinsics[cam]==np.eye(4)):  # 첫 수신만 저장
            self.meta.add_cam_intr(cam,msg.K)

    def _tf_cb(self,msg):
        for tf in msg.transforms:
            child=tf.child_frame_id.upper()
            ts=tf.header.stamp.to_sec()
            trans=[tf.transform.translation.x,tf.transform.translation.y,tf.transform.translation.z]
            rot=[tf.transform.rotation.w,tf.transform.rotation.x,tf.transform.rotation.y,tf.transform.rotation.z]
            mat=self.meta.tfmat(rot,trans)
            if child in CAM_KEYS:
                self.cam_tf[child].add(ts,mat)
            elif child=='LIDAR':
                self.lidar_tf.add(ts,mat)
            elif child=='EGO_POSE':
                self.ego_tf.add(ts,mat)

    def _lidar_cb(self,msg):
        stamp=msg.header.stamp.to_sec()
        pts=self._pc2_to_xyzi(msg)
        e2g,_=self.ego_tf.get_closest(stamp,0.1)
        l2e,_=self.lidar_tf.get_closest(stamp,0.1)
        self.lidar.add(pts,l2e,e2g,stamp)
        # extrinsics 최신화
        for cam in CAM_KEYS:
            mat,_=self.cam_tf[cam].get_closest(stamp,0.1)
            if mat is not None: self.meta.cam_extrinsics[cam]=mat
        if l2e is not None: self.meta.lidar_extrinsics=l2e
        # stamp 큐 push (최신만 유지)
        while not self.work_q.empty():
            try: self.work_q.get_nowait()
            except queue.Empty: break
        self.work_q.put(stamp)

    # ---------------- Worker ----------------
    def _worker(self):
        while not rospy.is_shutdown():
            start_time = time.time()
            stamp=self.work_q.get()
            self._infer(stamp)
            elapsed_time = time.time() - start_time
            self.count += 1
            self.total_time += elapsed_time

            rospy.loginfo(f"Average inference time: {self.total_time / self.count:.6f} seconds, fps: {1 / (self.total_time / self.count):.2f}")

    # ---------------- Inference -------------
    def _infer(self, stamp):
        imgs, aug=self.cams.fetch(stamp,0.1)
        self.meta.all_img_aug_matrix=aug
        meta_dict=self.meta.as_dict()
        pts=self.lidar.build()
        batch={'points':pts,'imgs':imgs}
        sample=Det3DDataSample()
        sample.set_metainfo(meta_dict)
        with amp.autocast(dtype=torch.float16):
            with torch.no_grad():
                res=self.model.predict(batch, batch_data_samples=[sample])
        # with torch.no_grad():
        #     res=self.model.predict(batch, batch_data_samples=[sample])
        pred=res[0].pred_instances_3d
        if pred is None or pred.bboxes_3d.tensor.shape[0]==0: return
        mask=pred.scores_3d>0.05
        boxes=pred.bboxes_3d[mask].tensor.cpu().numpy()
        scores=pred.scores_3d[mask].cpu().numpy().reshape(-1,1)
        labels=pred.labels_3d[mask].cpu().numpy().reshape(-1,1)
        flat=np.concatenate([boxes,scores,labels],1).flatten()
        det=DetectionResult()
        det.header.stamp=rospy.Time.from_sec(stamp)
        det.header.frame_id='lidar'
        det.result.data=flat.tolist()
        self.det_pub.publish(det)

        # pts_xy = batch['points'][0][:, :2].cpu().numpy()
        # self.plot_bev(
        #     points_xy=pts_xy,
        #     bboxes_3d=pred.bboxes_3d[mask],
        #     scores_3d=scores,
        #     class_ids=labels,
        #     out_path=f'./tmp/bev_ros_result_{self.count}.png',
        #     range_m=80.0
        # )
        self.lidar_pub.publish(self._create_pc2(pts[0], stamp))

    # ---------------- Utils -----------------
    def _pc2_to_xyzi(self,msg):
        step=msg.point_step
        pts=np.frombuffer(msg.data,np.float32).reshape(-1,4)
        return torch.from_numpy(pts.copy()).cuda()
    
    def _create_pc2(self, pts, stamp):
        header=Header()
        header.stamp=rospy.Time.from_sec(stamp)
        header.frame_id='lidar'
        pts = pts.float()
        pc2=PointCloud2()
        pc2.header=header
        pc2.height=1
        pc2.width=pts.shape[0]
        pc2.fields=[PointField('x',0,PointField.FLOAT32,1),
                    PointField('y',4,PointField.FLOAT32,1),
                    PointField('z',8,PointField.FLOAT32,1),
                    PointField('intensity',12,PointField.FLOAT32,1),
                    PointField('time',16,PointField.FLOAT32,1),]
        pc2.is_bigendian=False
        pc2.point_step=20
        pc2.row_step=pc2.point_step*pc2.width
        pc2.is_dense=True
        pc2.data=pts.cpu().numpy().tobytes()
        return pc2

    def spin(self):
        rospy.spin()

    def plot_bev(self, points_xy, bboxes_3d, scores_3d=None, class_ids=None,
                 out_path='bev_result.png', range_m=50.0):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(points_xy[:, 0], points_xy[:, 1], c='b', s=1, alpha=0.3)

        boxes = bboxes_3d.corners.cpu().numpy()
        for i in range(boxes.shape[0]):
            box = boxes[i]
            for j in range(4):
                ax.plot([box[j, 0], box[j+4, 0]], [box[j, 1], box[j+4, 1]], 'r', linewidth=0.5)
                ax.plot([box[j, 0], box[(j+1) % 4, 0]], [box[j, 1], box[(j+1) % 4, 1]], 'r', linewidth=0.5)
                ax.plot([box[j+4, 0], box[(j+1) % 4+4, 0]], [box[j+4, 1], box[(j+1) % 4+4, 1]], 'r', linewidth=0.5)
            if scores_3d is not None:
                ax.text(box[0, 0], box[0, 1], f"{scores_3d[i,0]}", color='yellow', fontsize=8)
            if class_ids is not None:
                ax.text(box[2, 0], box[2, 1], f"{class_ids[i,0]}", color='green', fontsize=8)

        ax.set_xlim(-range_m, range_m)
        ax.set_ylim(-range_m, range_m)
        ax.set_aspect('equal')
        ax.set_title("BEV Visualization")
        plt.savefig(out_path)
        plt.close()
        rospy.loginfo(f"BEV 시각화 결과 저장: {out_path}")

if __name__=='__main__':
    InferenceNode().spin()
