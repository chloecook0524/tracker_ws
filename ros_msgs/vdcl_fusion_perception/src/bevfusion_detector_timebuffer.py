#!/usr/bin/env python

import sys
import rospy
import torch
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from pyquaternion import Quaternion
from collections import deque
import matplotlib.pyplot as plt

from sensor_msgs.msg import PointCloud2, Image, CameraInfo, CompressedImage, PointField
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, Float32MultiArray
from vdcl_fusion_perception.msg import DetectionResult

from mmengine import Config
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
import time


# --------------------------------------------------------------------------------
# Helper classes for buffering data by timestamp and matching to a desired time
# --------------------------------------------------------------------------------
class TimeBuffer:
    """
    Stores (stamp, data) up to maxlen. Provides method to get closest stamp.
    stamp should be a float (e.g. time in seconds).
    """
    def __init__(self, maxlen=20):
        self.buffer = deque(maxlen=maxlen)

    def add(self, stamp, data):
        self.buffer.append((stamp, data))

    def get_closest(self, target_stamp, max_dt=0.2):
        """
        Return the data whose stamp is closest to target_stamp.
        If the closest difference is greater than max_dt, return None.
        """
        best_data = None
        best_stamp = None
        min_diff = float('inf')
        min_idx = -1
        for i, (s, d) in enumerate(self.buffer):
            diff = abs(s - target_stamp)
            if diff < min_diff:
                min_diff = diff
                best_data = d
                best_stamp = s
                min_idx = i

        if min_diff <= max_dt:
            return best_data, best_stamp
        else:
            print(f"Closest stamp {best_stamp} is too far from target {target_stamp} (diff={min_diff:.3f}, index={min_idx})")
            # print(f"Buffer: {[f'{s:.3f}' for s, _ in self.buffer]}")
            return None, None


# --------------------------------------------------------------------------------
# Original classes from your code, now slightly modified to handle time-based data
# --------------------------------------------------------------------------------
CAM_KEYS = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]

class Meta:
    def __init__(self, cam_keys):
        self.cam_keys = cam_keys
        self.cam_intrinsics = {}
        self.cam_extrinsics = {}
        self.all_img_aug_matrix = None
        self.lidar_extrinsics = None
        self.sample_idx = 0

        for cam_key in cam_keys:
            self.cam_extrinsics[cam_key] = np.eye(4, dtype=np.float32)
            self.cam_intrinsics[cam_key] = np.eye(4, dtype=np.float32)
        self.all_img_aug_matrix = np.repeat([np.eye(4, dtype=np.float32)],6)
        self.lidar_extrinsics = np.eye(4, dtype=np.float32)

    def get_transform_matrix(self, rotation, translation):
        transform = np.eye(4, dtype=np.float32)
        R = Quaternion(rotation).rotation_matrix  # (3,3)
        transform[:3, :3] = R
        transform[:3, 3] = translation
        return transform

    def add_camera_intrinsics(self, cam_name, intrinsics):
        intrinsics = np.array(intrinsics).reshape(3, 3)
        intrinsics = np.block([[intrinsics, np.zeros((3, 1))], [0, 0, 0, 1]])
        self.cam_intrinsics[cam_name] = intrinsics

    def get_meta_info(self):
        all_lidar2cam = []
        all_cam2lidar = []
        all_cam2img = []
        all_lidar2img = []

        for cam_name in self.cam_keys:
            # If extrinsics for this camera are not set yet, skip
            if cam_name not in self.cam_extrinsics or self.lidar_extrinsics is None:
                # Return None if incomplete
                return None

            lidar2cam = np.linalg.inv(self.cam_extrinsics[cam_name]) @ self.lidar_extrinsics
            cam2lidar = np.linalg.inv(lidar2cam)
            cam2img = self.cam_intrinsics.get(cam_name, None)
            if cam2img is None:
                return None

            lidar2img = cam2img @ lidar2cam

            all_lidar2cam.append(lidar2cam)
            all_cam2lidar.append(cam2lidar)
            all_cam2img.append(cam2img)
            all_lidar2img.append(lidar2img)

        meta_info = {
            'sample_idx': self.sample_idx,
            'lidar2cam': np.stack(all_lidar2cam),
            'cam2lidar': np.stack(all_cam2lidar),
            'cam2img': np.stack(all_cam2img),
            'lidar2img': np.stack(all_lidar2img),
            'img_aug_matrix': self.all_img_aug_matrix,
            'num_pts_feats': 5,
            'box_type_3d': LiDARInstance3DBoxes,
        }
        self.sample_idx += 1
        self.sample_idx %= 10000
        return meta_info

class LiDARPointCloud:
    def __init__(self, num_sweeps=10):
        self.num_sweeps = num_sweeps
        self.points = deque(maxlen=num_sweeps)
        self.ego2globals = deque(maxlen=num_sweeps)
        self.lidar2ego = deque(maxlen=num_sweeps)
        self.timestamps = deque(maxlen=num_sweeps)

    def add_points(self, points, lidar2ego, ego2global, timestamp):
        self.points.append(points.cuda())
        if ego2global is not None:
            self.ego2globals.append(torch.from_numpy(ego2global).cuda())
        else:
            self.ego2globals.append(None)

        if lidar2ego is not None:
            self.lidar2ego.append(torch.from_numpy(lidar2ego).cuda())
        else:
            self.lidar2ego.append(None)
        self.timestamps.append(timestamp)

    def get_points(self):
        """
        Returns a single list of shape [N, 5] as your code
        does time-compensation for multi-sweeps.
        """
        latest_points = self.points[-1]
        zeros_time = torch.zeros((latest_points.shape[0], 1), device='cuda')
        points_with_time = torch.cat([latest_points, zeros_time], dim=1)
        compensated_points = [points_with_time]
        if self.num_sweeps == 1:
            return compensated_points

        ego2global_current = self.ego2globals[-1]
        global2ego_current = torch.inverse(ego2global_current)
        lidar2ego_current = self.lidar2ego[-1]
        ego2lidar_current = torch.inverse(lidar2ego_current)
        timestamp_current = self.timestamps[-1]

        # fuse older sweeps with time compensation
        for i in range(len(self.points) - 1):
            if timestamp_current - self.timestamps[i] > 0.6:
                continue
            elif self.ego2globals[i] is None or self.lidar2ego[i] is None:
                continue

            points_i = self.points[i]
            ego2global_i = self.ego2globals[i]
            lidar2ego_i = self.lidar2ego[i]
            timestamp_i = self.timestamps[i]

            xyz = points_i[:, :3]
            intensity = points_i[:, 3:4]
            points_h = torch.cat([xyz, torch.ones((xyz.shape[0], 1), device='cuda')], dim=1)

            points_global = (points_h @ lidar2ego_i.T) @ ego2global_i.T
            points_compensated = (points_global @ global2ego_current.T) @ ego2lidar_current.T
            points_compensated = points_compensated[:, :3]

            time_diff = timestamp_current - timestamp_i
            time_channel = torch.full((points_i.shape[0], 1), fill_value=time_diff, device='cuda')

            compensated = torch.cat([points_compensated, intensity, time_channel], dim=1)
            compensated_points.append(compensated)

        all_points = torch.cat(compensated_points, dim=0)
        torch.cuda.synchronize()
        return [all_points]

class CameraImage:
    """
    Holds multiple camera images for all 6 cameras, matched by time later.
    For final shape (256, 704) as in your code.
    """
    def __init__(self, cam_keys, img_shape=(256, 704)):
        self.cam_keys = cam_keys
        self.img_shape = img_shape

        # Instead of storing a single image, let's store a buffer per camera.
        self.img_buffers = {k: TimeBuffer(maxlen=200) for k in cam_keys}

    def add_img(self, img, cam_key, stamp):
        # Store in the buffer for the given camera
        self.img_buffers[cam_key].add(stamp, img)

    def apply_img_transform(self, img, final_dim=(256, 704)):
        _, h, w = img.shape
        final_h, final_w = final_dim
        resize = final_w / w
        new_w = int(w * resize)
        new_h = int(h * resize)
        crop_h = int(new_h - final_h)

        # Resize
        img_resized = TF.resize(img, [new_h, new_w], antialias=True)

        # Crop
        img_cropped = TF.crop(img_resized, crop_h, 0, final_h, final_w)

        # Aug matrix
        rotation = np.eye(2, dtype=np.float32) * resize
        translation = np.array([0, -crop_h], dtype=np.float32)
        img_aug_matrix = np.eye(4, dtype=np.float32)
        img_aug_matrix[:2, :2] = rotation
        img_aug_matrix[:2, 3] = translation
        return img_cropped, img_aug_matrix

    def get_matched_images_and_aug(self, target_stamp, max_dt=0.2):
        """
        For each camera, find the closest image to target_stamp
        and return a stacked image tensor plus augmentation matrix.
        If any camera is missing an image within max_dt, returns None.
        """
        all_imgs = []
        all_aug_matrices = []
        for k in self.cam_keys:
            img, stamp_found = self.img_buffers[k].get_closest(target_stamp, max_dt=max_dt)
            # img = None
            if img is None:
                dummy_img = torch.zeros((3, *self.img_shape), device='cuda')
                all_imgs.append(dummy_img)  # Placeholder for missing camera
                all_aug_matrices.append(np.eye(4, dtype=np.float32))  # Identity for missing camera
                # return None, None  # Some camera missing data
            else:
                # Apply transforms here
                transformed_img, aug = self.apply_img_transform(img)
                all_imgs.append(transformed_img.float())
                all_aug_matrices.append(aug)

        imgs_tensor = torch.stack(all_imgs, dim=0)  # [N, 3, H, W]
        imgs_tensor = imgs_tensor.unsqueeze(0)     # [1, N, 3, H, W]

        return imgs_tensor, np.stack(all_aug_matrices)

class InferenceNode:
    def __init__(self):
        rospy.init_node("inference_node")

        self.sample_index = 0

        self.model = self.load_model()
        self.model.eval()

        # This holds camera intrinsics, extrinsics, etc.
        self.meta_info = Meta(CAM_KEYS)

        # Buffers for camera info (intrinsics) – typically doesn't change, so just store once
        self.camera_info_received = set()

        # Buffer for TF transformations: we only need ego2global here,
        # but store them in a ring buffer keyed by time to pick nearest
        self.ego_tf_buffer = TimeBuffer(maxlen=200)

        # A separate dictionary for camera extrinsics if these can vary over time.
        # Often these extrinsics are static, but let's keep it flexible.
        self.cam_tf_buffers = {k: TimeBuffer(maxlen=200) for k in CAM_KEYS}
        self.lidar_tf_buffer = TimeBuffer(maxlen=200)

        self.camera_images = CameraImage(CAM_KEYS, img_shape=(256, 704))
        self.pointcloud = LiDARPointCloud(num_sweeps=10)

        # Subscribe to camera image topics
        for cam in CAM_KEYS:
            rospy.Subscriber(cam.lower(), Image, self.image_callback, callback_args=cam, queue_size=1)
            # rospy.Subscriber(f"{cam.lower()}/compressed", CompressedImage, self.image_callback, callback_args=cam)
            rospy.Subscriber(f"{cam.lower()}_info", CameraInfo, self.camera_info_callback, callback_args=cam, queue_size=1, buff_size=2**24)

        rospy.Subscriber("tf", TFMessage, self.tf_callback, queue_size=1)
        rospy.Subscriber("lidar_points", PointCloud2, self.lidar_callback, queue_size=1, buff_size=2**24)
        # rospy.Subscriber("/velodyne_FC/velodyne_points", PointCloud2, self.lidar_callback, queue_size=1)

        # self.merged_point_pub = rospy.Publisher("/merged_points", PointCloud2, queue_size=1)

        self.detection_pub = rospy.Publisher("/detection_results", DetectionResult, queue_size=1)

    def load_model(self):
        cfg = Config.fromfile('/home/sgsp/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0100_depth_with_lidar_depth.py')
        model = MODELS.build(cfg.model)
        load_checkpoint(model,
                        '/home/sgsp/mmdetection3d/work_dirs/bevfusion_lidar-cam_voxel0100_depth_with_lidar_depth_400q/epoch_6.pth',
                        map_location='cpu')
        
        # cfg = Config.fromfile('/home/sgsp/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0100_depth_from_scratch.py')
        # model = MODELS.build(cfg.model)
        # load_checkpoint(model,
        #                 '/home/sgsp/mmdetection3d/work_dirs/bevfusion_lidar-cam_voxel0100_depth_from_scratch/epoch_6.pth',
        #                 map_location='cpu')
        return model.cuda()

    # ----------------------------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------------------------
    def image_callback(self, msg, cam):
        try:
            cv_img, timestamp = self.image_msg_to_image(msg)
            self.camera_images.add_img(cv_img, cam, timestamp)
        except Exception as e:
            rospy.logwarn(f"Failed to convert image from {cam}: {e}")

    def camera_info_callback(self, msg, cam):
        # Typically camera info doesn't change, so we store once
        if cam not in self.camera_info_received:
            self.meta_info.add_camera_intrinsics(cam, msg.K)
            self.camera_info_received.add(cam)

    def tf_callback(self, msg):
        """
        We store TF transforms in ring buffers, keyed by stamp. Then we pick the
        best match in time during the LiDAR callback. If your extrinsics are truly static,
        you might not need time-based matching for camera extrinsics, but it’s shown here.
        """
        for transform_stamped in msg.transforms:
            # Frame data
            child_frame_id = transform_stamped.child_frame_id.upper()  # e.g. 'CAM_FRONT', 'lidar', etc.
            stamp_sec = transform_stamped.header.stamp.to_sec()

            t = np.array([transform_stamped.transform.translation.x,
                          transform_stamped.transform.translation.y,
                          transform_stamped.transform.translation.z])
            r = np.array([transform_stamped.transform.rotation.w,
                          transform_stamped.transform.rotation.x,
                          transform_stamped.transform.rotation.y,
                          transform_stamped.transform.rotation.z])
            mat = self.meta_info.get_transform_matrix(r, t)

            if child_frame_id in CAM_KEYS:
                self.cam_tf_buffers[child_frame_id].add(stamp_sec, mat)
            elif child_frame_id == "LIDAR":
                self.lidar_tf_buffer.add(stamp_sec, mat)
            elif child_frame_id == "EGO_POSE":
                self.ego_tf_buffer.add(stamp_sec, mat)

    def lidar_callback(self, msg):
        callback_start = time.time()
        """
        When a new pointcloud arrives, we:
        1. Find the ego2global closest in time
        2. Find camera extrinsics & intrinsics
        3. Find images for each cam that are closest in time
        Then run inference.
        """
        pc_stamp = msg.header.stamp.to_sec()
        # 1) Get the best match for ego2global
        ego2global, _ = self.ego_tf_buffer.get_closest(pc_stamp, max_dt=0.1)
        if ego2global is None:
            rospy.logwarn("No matching ego pose transform found for time %.3f" % pc_stamp)
            return

        # 2) For each camera extrinsic, pick the closest
        for cam in CAM_KEYS:
            mat, cam_stamp = self.cam_tf_buffers[cam].get_closest(pc_stamp, max_dt=0.1)
            if mat is not None:
                self.meta_info.cam_extrinsics[cam] = mat

        # 3) For the LiDAR extrinsic
        lidar2ego, _ = self.lidar_tf_buffer.get_closest(pc_stamp, max_dt=0.1)
        if lidar2ego is not None:
            self.meta_info.lidar_extrinsics = lidar2ego

        # 4) Now add LiDAR points
        points_lidar, timestamp = self.pointcloud2_to_xyzi(msg)
        self.pointcloud.add_points(points_lidar, lidar2ego, ego2global, timestamp)

        # 5) Match camera images to this time
        imgs_tensor, aug_mats = self.camera_images.get_matched_images_and_aug(pc_stamp, max_dt=0.1)
        if imgs_tensor is None:
            rospy.logwarn("Not all cameras had a matching image for time %.3f" % pc_stamp)
            return

        # 6) Build final meta_info
        self.meta_info.all_img_aug_matrix = aug_mats
        meta_dict = self.meta_info.get_meta_info()
        if meta_dict is None:
            rospy.logwarn("Meta info incomplete (missing intringhsics/extrinsics). Skipping.")
            return

        # 7) Prepare final input for model
        st_time = time.time()
        points = self.pointcloud.get_points()

        # self.merged_point_pub.publish(self.numpy_to_pointcloud2(points[0].cpu().numpy(), frame_id='lidar'))

        batch_inputs_dict = {
            'points': points,
            'imgs': imgs_tensor,
        }

        data_sample = Det3DDataSample()
        data_sample.set_metainfo(meta_dict)

        # 8) Inference
        st_time = time.time()   
        with torch.no_grad():
            results = self.model.predict(batch_inputs_dict, batch_data_samples=[data_sample])
        et_time = time.time()
        rospy.loginfo(f"Inference time: {et_time - st_time:.3f} seconds")
        
        # 9) Post-processing and visualization
        pred_3d = results[0].pred_instances_3d
        if pred_3d is None or pred_3d.bboxes_3d.tensor.shape[0] == 0:
            rospy.loginfo("No predicted 3D boxes.")
            return

        bboxes_3d = pred_3d.bboxes_3d
        scores_3d = pred_3d.scores_3d
        labels_3d = pred_3d.labels_3d

        mask = scores_3d > 0.05
        bboxes_3d = bboxes_3d[mask]
        scores_3d = scores_3d[mask].cpu().numpy().reshape(-1, 1)
        labels_3d = labels_3d[mask].cpu().numpy().reshape(-1, 1)

        # print(f"bboxes_3d.shape: {bboxes_3d.tensor.shape}")
        # print(f"scores_3d.shape: {scores_3d.shape}")
        # print(f"labels_3d.shape: {labels_3d.shape}")

        # bboxes_3d = x,y,z,x_size,y_size,z_size,yaw,vx,vy
        bboxes_tensor_3d = bboxes_3d.tensor.cpu().numpy()

        detection_result = DetectionResult()
        result_3d = np.concatenate([bboxes_tensor_3d, scores_3d, labels_3d], axis=1)
        detection_result.result.data = result_3d.flatten().tolist()
        detection_result.header.stamp = msg.header.stamp
        detection_result.header.frame_id = "lidar"
        self.detection_pub.publish(detection_result)
        print("delay:", rospy.Time.now().to_sec() - msg.header.stamp.to_sec())

        # pts_xy = batch_inputs_dict['points'][0][:, :2].cpu().numpy()
        # self.plot_bev(
        #     points_xy=pts_xy,
        #     bboxes_3d=bboxes_3d,
        #     scores_3d=scores_3d,
        #     class_ids=labels_3d,
        #     out_path=f'./tmp/bev_ros_result_{self.sample_index}_{pc_stamp}.png',
        #     range_m=80.0
        # )

        # self.visualize_6cams_with_boxes(
        #     cam_imgs=imgs_tensor[0],
        #     bboxes_3d=bboxes_3d,
        #     lidar2img_mats=meta_dict['lidar2img'],
        #     img_aug_mats=meta_dict['img_aug_matrix'],
        #     out_path=f'./tmp/cam6_ros_result_{self.sample_index}_{cam_stamp}.png'
        # )
        
        self.sample_index += 1
        rospy.loginfo(f"Sample index: {self.sample_index}")
        self.sample_index %= 10000

        print(f"Total callback time: {time.time() - callback_start:.3f} seconds")


    # ----------------------------------------------------------------------------
    # Utility Methods
    # ----------------------------------------------------------------------------
    def pointcloud2_to_xyzi(self, pc2_msg):
        points = np.frombuffer(pc2_msg.data, dtype=np.float32)
        points = points.reshape((pc2_msg.width, 4))
        points = torch.from_numpy(points).cuda()
        timestamp = pc2_msg.header.stamp.to_sec()
        return points, timestamp

    def image_msg_to_image(self, img_msg):
        if isinstance(img_msg, CompressedImage):
            img = np.frombuffer(img_msg.data, dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        else:
            img = np.frombuffer(img_msg.data, dtype=np.uint8)
            img = img.reshape((img_msg.height, img_msg.width, 3))
        img = img[..., ::-1].copy()  # BGR to RGB
        img = torch.from_numpy(img).cuda()
        img = img.permute(2, 0, 1)
        timestamp = img_msg.header.stamp.to_sec()
        return img, timestamp
    
    def numpy_to_pointcloud2(self, points, frame_id='lidar'):
        pc2_msg = PointCloud2()
        pc2_msg.header.frame_id = frame_id
        pc2_msg.height = 1
        pc2_msg.width = points.shape[0]
        pc2_msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='time', offset=16, datatype=PointField.FLOAT32, count=1),
        ]
        pc2_msg.is_bigendian = False
        pc2_msg.point_step = 20
        pc2_msg.row_step = pc2_msg.point_step * points.shape[0]
        pc2_msg.is_dense = True

        pc2_msg.data = points.astype(np.float32).tobytes()
        return pc2_msg

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

    def visualize_6cams_with_boxes(self, cam_imgs, bboxes_3d,
                                   lidar2img_mats, img_aug_mats,
                                   out_path='cam_6view.png'):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        axes = axes.flat
        corners_all = bboxes_3d.corners.cpu().numpy()  # (N, 8, 3)

        for i in range(6):
            img = cam_imgs[i].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            axes[i].imshow(img)

            for box_idx in range(corners_all.shape[0]):
                corners = corners_all[box_idx]
                box = np.concatenate([corners, np.ones((8, 1))], axis=1)
                box = box @ lidar2img_mats[i].T
                box[:, :2] /= box[:, 2:3]
                if (box[:, 2] <= 0).any():
                    continue
                box = box @ img_aug_mats[i].T

                for k in range(4):
                    axes[i].plot([box[k, 0], box[k+4, 0]],
                                 [box[k, 1], box[k+4, 1]], 'r', linewidth=0.3)
                    axes[i].plot([box[k, 0], box[(k+1) % 4, 0]],
                                 [box[k, 1], box[(k+1) % 4, 1]], 'r', linewidth=0.3)
                    axes[i].plot([box[k+4, 0], box[(k+1) % 4+4, 0]],
                                 [box[k+4, 1], box[(k+1) % 4+4, 1]], 'r', linewidth=0.3)

            axes[i].set_xlim(0, img.shape[1])
            axes[i].set_ylim(img.shape[0], 0)
            axes[i].set_title(f"Cam {i}")

        fig.suptitle("6-Camera View with Projected BBoxes")
        fig.tight_layout()
        plt.savefig(out_path)
        plt.close()
        rospy.loginfo(f"6카메라 시각화 결과 저장: {out_path}")

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    node = InferenceNode()
    node.spin()
