# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Any, Dict, Tuple

import mmcv
import numpy as np
from mmengine.fileio import get

from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class BEVLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.

    ``BEVLoadMultiViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        num_ref_frames (int): Number of frame in loading. Defaults to -1.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # TODO: consider split the multi-sweep part out of this pipeline
        # Derive the mask and transform for loading of multi-sweep data
        if self.num_ref_frames > 0:
            # init choice with the current frame
            init_choice = np.array([0], dtype=np.int64)
            num_frames = len(results['img_filename']) // self.num_views - 1
            if num_frames == 0:  # no previous frame, then copy cur frames
                choices = np.random.choice(
                    1, self.num_ref_frames, replace=True)
            elif num_frames >= self.num_ref_frames:
                # NOTE: suppose the info is saved following the order
                # from latest to earlier frames
                if self.test_mode:
                    choices = np.arange(num_frames - self.num_ref_frames,
                                        num_frames) + 1
                # NOTE: +1 is for selecting previous frames
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=False) + 1
            elif num_frames > 0 and num_frames < self.num_ref_frames:
                if self.test_mode:
                    base_choices = np.arange(num_frames) + 1
                    random_choices = np.random.choice(
                        num_frames,
                        self.num_ref_frames - num_frames,
                        replace=True) + 1
                    choices = np.concatenate([base_choices, random_choices])
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=True) + 1
            else:
                raise NotImplementedError
            choices = np.concatenate([init_choice, choices])
            select_filename = []
            for choice in choices:
                select_filename += results['img_filename'][choice *
                                                           self.num_views:
                                                           (choice + 1) *
                                                           self.num_views]
            results['img_filename'] = select_filename
            for key in ['cam2img', 'lidar2cam']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += results[key][choice *
                                                       self.num_views:(choice +
                                                                       1) *
                                                       self.num_views]
                    results[key] = select_results
            for key in ['ego2global']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += [results[key][choice]]
                    results[key] = select_results
            # Transform lidar2cam to
            # [cur_lidar]2[prev_img] and [cur_lidar]2[prev_cam]
            for key in ['lidar2cam']:
                if key in results:
                    # only change matrices of previous frames
                    for choice_idx in range(1, len(choices)):
                        pad_prev_ego2global = np.eye(4)
                        prev_ego2global = results['ego2global'][choice_idx]
                        pad_prev_ego2global[:prev_ego2global.
                                            shape[0], :prev_ego2global.
                                            shape[1]] = prev_ego2global
                        pad_cur_ego2global = np.eye(4)
                        cur_ego2global = results['ego2global'][0]
                        pad_cur_ego2global[:cur_ego2global.
                                           shape[0], :cur_ego2global.
                                           shape[1]] = cur_ego2global
                        cur2prev = np.linalg.inv(pad_prev_ego2global).dot(
                            pad_cur_ego2global)
                        for result_idx in range(choice_idx * self.num_views,
                                                (choice_idx + 1) *
                                                self.num_views):
                            results[key][result_idx] = \
                                results[key][result_idx].dot(cur2prev)
        # Support multi-view images with different shapes
        # TODO: record the origin shape and padded shape
        filename, cam2img, lidar2cam, cam2lidar, lidar2img = [], [], [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            lidar2cam.append(cam_item['lidar2cam'])

            lidar2cam_array = np.array(cam_item['lidar2cam']).astype(
                np.float32)
            lidar2cam_rot = lidar2cam_array[:3, :3]
            lidar2cam_trans = lidar2cam_array[:3, 3:4]
            camera2lidar = np.eye(4)
            camera2lidar[:3, :3] = lidar2cam_rot.T
            camera2lidar[:3, 3:4] = -1 * np.matmul(
                lidar2cam_rot.T, lidar2cam_trans.reshape(3, 1))
            cam2lidar.append(camera2lidar)

            cam2img_array = np.eye(4).astype(np.float32)
            cam2img_array[:3, :3] = np.array(cam_item['cam2img']).astype(
                np.float32)
            cam2img.append(cam2img_array)
            lidar2img.append(cam2img_array @ lidar2cam_array)

        results['img_path'] = filename
        results['cam2img'] = np.stack(cam2img, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0)
        results['cam2lidar'] = np.stack(cam2lidar, axis=0)
        results['lidar2img'] = np.stack(lidar2img, axis=0)

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        imgs = [
            mmcv.imfrombytes(
                img_byte,
                flag=self.color_type,
                backend='pillow',
                channel_order='rgb') for img_byte in img_bytes
        ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames

        # visualize
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(2, 3, figsize=(12, 6))
        # for i in range(len(results['img'])):
        #     axs[i//3, i%3].imshow(results['img'][i]/255)
        #     axs[i//3, i%3].set_title('Image '+str(i))
        # plt.savefig('test_data/img_ori'+str(results['sample_idx'])+'.png')
        # plt.close()

        return results


@TRANSFORMS.register_module()
class LoadBEVSegmentation:
    def __init__(
        self,
        dataset_root: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        classes: Tuple[str, ...],
    ) -> None:
        super().__init__()
        
        self.dataset_root = dataset_root
        self.ori_size = [[-51.2,51.2,0.2],[-51.2,51.2,0.2]]
        self.target_size = [xbound, ybound]
        assert self.ori_size == self.target_size, "change the size of the map are not implemented"
        self.classes = classes
        self.class_labels = {    
            "pedestrian": 0,
            "traffic cone": 1,    
            "motorcycle": 2,
            "bicycle": 3,
            "car": 4,
            "truck": 5,
            "bus": 6,
            "construction_vehicle": 7,
            "trailer": 8,
            "other_object": 9,
            "barrier": 10,
            "static_object": 11,
            "sidewalk": 12,
            "other_flat": 13,
            "vegetation": 14,
            "drivable_surface": 15,
            "unknown" : 16,
        }
        assert len(self.class_labels) == len(self.classes) + 1, "seg classes are predefined"

        self.color_map = {
            0: [0,0,0.90196078],
            1: [0.18431373,0.30980392,0.30980392],
            2: [0.43921569,0.50196078,0.56470588],
            3: [1,0.23921569,0.38823529],
            4: [0.8627451,0.07843137,0.23529412],
            5: [1,0.61960784,0],
            6: [1,0.38823529,0.27843137],
            7: [1,0.27058824,0],
            8: [0.91372549,0.58823529,0.2745098],
            9: [1,0.54901961,0],
            10: [0.41176471,0.41176471,0.41176471],
            11: [0.87058824,0.72156863,0.52941176],
            12: [0,0.68627451,0],
            13: [0.29411765,0,0.29411765],
            14: [0.43921569,0.70588235,0.23529412],
            15: [0,0.81176471,0.74901961],
            16: [1,1,1]
        }

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:       
        lidar_token = data['lidar_token']
        map_x = int((self.ori_size[0][1]-self.ori_size[0][0])/self.ori_size[0][2])
        map_y = int((self.ori_size[1][1]-self.ori_size[1][0])/self.ori_size[1][2])
        occ_bev = np.fromfile(self.dataset_root + '/' + lidar_token + '.bin', dtype=np.int16).reshape(map_x,map_y)
        # occ_bev = np.transpose(occ_bev, (1,0))

        # lidar2point = data["lidar_aug_matrix"]
        # point2lidar = np.linalg.inv(lidar2point)
        # lidar2point_2d = np.vstack([lidar2point[:2,[0,1,3]],[0,0,1]])
        # # print("lidar2point_2d:",lidar2point_2d)
        # point2lidar_2d = np.vstack([point2lidar[:2,[0,1,3]],[0,0,1]])
        # # print("point2lidar_2d:",point2lidar_2d)

        # bev2lidar = np.array([[1,0,self.ori_size[0][0]/self.ori_size[0][2]],[0,1,self.ori_size[1][0]/self.ori_size[1][2]],[0,0,1]])
        # lidar2bev = np.array([[1,0,-self.ori_size[0][0]/self.ori_size[0][2]],[0,1,-self.ori_size[1][0]/self.ori_size[1][2]],[0,0,1]])
        # # print(lidar2bev)

        # point2bev = lidar2bev @ lidar2point_2d @ bev2lidar
        # point2bev = point2bev[:2,:]
        # # print("point2bev:",point2bev)

        # occ_bev_aug = cv2.warpAffine(occ_bev, point2bev, (map_x,map_y), flags=cv2.INTER_NEAREST, borderValue=16)

        # data["gt_seg_map"] = occ_bev_aug
        data["gt_seg_map"] = occ_bev

        # #visualize
        # import matplotlib.pyplot as plt
        # import matplotlib.colors as mcolors
        # cmap = mcolors.ListedColormap([self.color_map[i] for i in range(17)])
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # im=ax.imshow(occ_bev, cmap=cmap, interpolation='none')
        # ax.invert_yaxis()
        # fig.colorbar(im, ax=ax)
        # plt.savefig('test_data/bev_ori'+str(data['sample_idx'])+'.png')
        # plt.close()
        
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # im=ax.imshow(occ_bev_aug, cmap=cmap, interpolation='none')
        # ax.invert_yaxis()
        # fig.colorbar(im, ax=ax)
        # plt.savefig('test_data/bev_aug'+str(data['sample_idx'])+'.png')
        # plt.close()

        return data