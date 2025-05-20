# modify from https://github.com/mit-han-lab/bevfusion
from typing import Any, Dict

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from PIL import Image
import cv2

from mmdet3d.datasets import GlobalRotScaleTrans
from mmdet3d.registry import TRANSFORMS

from mmdet3d.structures.ops import box_np_ops


@TRANSFORMS.register_module()
class ImageAug3D(BaseTransform):

    def __init__(self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip,
                 is_train):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results):
        H, W = results['ori_shape']
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, rotation, translation, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data['img']
        new_imgs = []
        transforms = []
        for img in imgs:
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                data)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(np.array(new_img).astype(np.float32))
            transforms.append(transform.numpy())
        data['img'] = new_imgs
        # update the calibration matrices
        data['img_aug_matrix'] = transforms
        return data


@TRANSFORMS.register_module()
class BEVFusionRandomFlip3D:
    """Compared with `RandomFlip3D`, this class directly records the lidar
    augmentation matrix in the `data`."""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flip_horizontal = np.random.choice([0, 1])
        flip_vertical = np.random.choice([0, 1])

        rotation = np.eye(3)
        if flip_horizontal:
            rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
            if 'points' in data:
                data['points'].flip('horizontal')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('horizontal')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, :, ::-1].copy()
            if 'gt_seg_map' in data:
                data['gt_seg_map'] = data['gt_seg_map'][::-1, :].copy()

        if flip_vertical:
            rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
            if 'points' in data:
                data['points'].flip('vertical')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('vertical')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, ::-1, :].copy()
            if 'gt_seg_map' in data:
                data['gt_seg_map'] = data['gt_seg_map'][:, ::-1].copy()

        if 'lidar_aug_matrix' not in data:
            data['lidar_aug_matrix'] = np.eye(4)
        data['lidar_aug_matrix'][:3, :] = rotation @ data[
            'lidar_aug_matrix'][:3, :]
        return data


@TRANSFORMS.register_module()
class BEVFusionGlobalRotScaleTrans(GlobalRotScaleTrans):
    """Compared with `GlobalRotScaleTrans`, the augmentation order in this
    class is rotation, translation and scaling (RTS)."""

    def transform(self, input_dict: dict) -> dict:
        """Private function to rotate, scale and translate bounding boxes and
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans' and `gt_bboxes_3d` are updated
            in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._trans_bbox_points(input_dict)
        self._scale_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'T', 'S'])

        lidar_augs = np.eye(4)
        lidar_augs[:3, :3] = input_dict['pcd_rotation'].T * input_dict[
            'pcd_scale_factor']
        lidar_augs[:3, 3] = input_dict['pcd_trans'] * \
            input_dict['pcd_scale_factor']

        if 'lidar_aug_matrix' not in input_dict:
            input_dict['lidar_aug_matrix'] = np.eye(4)
        input_dict[
            'lidar_aug_matrix'] = lidar_augs @ input_dict['lidar_aug_matrix']
        
        ## transform gt_seg_map
        if 'gt_seg_map' in input_dict:
            lidar2point_2d = np.vstack([lidar_augs[:2,[0,1,3]],[0,0,1]])

            occ_bev = input_dict['gt_seg_map']

            map_x = occ_bev.shape[0]
            map_y = occ_bev.shape[1]

            bev2lidar = np.array([[1,0,-map_x/2],[0,1,-map_y/2],[0,0,1]])
            lidar2bev = np.array([[1,0,map_x/2],[0,1,map_y/2],[0,0,1]])

            point2bev = lidar2bev @ lidar2point_2d @ bev2lidar
            point2bev = point2bev[:2,:]

            # occ_bev_aug = cv2.warpAffine(occ_bev, point2bev, (map_x,map_y), flags=cv2.INTER_NEAREST, borderValue=int(occ_bev.max()))
            occ_bev_aug = cv2.warpAffine(occ_bev, point2bev, (map_x,map_y), flags=cv2.INTER_NEAREST, borderValue=16)

            input_dict["gt_seg_map"] = occ_bev_aug

        return input_dict


@TRANSFORMS.register_module()
class GridMask(BaseTransform):

    def __init__(
        self,
        use_h,
        use_w,
        max_epoch,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=0,
        prob=1.0,
        fixed_prob=False,
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.epoch = None
        self.max_epoch = max_epoch
        self.fixed_prob = fixed_prob

    def set_epoch(self, epoch):
        self.epoch = epoch
        if not self.fixed_prob:
            self.set_prob(self.epoch, self.max_epoch)

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * self.epoch / self.max_epoch

    def transform(self, results):
        if np.random.rand() > self.prob:
            return results
        imgs = results['img']
        h = imgs[0].shape[0]
        w = imgs[0].shape[1]
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.length = np.random.randint(1, d)
        else:
            self.length = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.length, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.length, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h,
                    (ww - w) // 2:(ww - w) // 2 + w]

        mask = mask.astype(np.float32)
        mask = mask[:, :, None]
        if self.mode == 1:
            mask = 1 - mask

        # mask = mask.expand_as(imgs[0])
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            imgs = [x * mask + offset for x in imgs]
        else:
            imgs = [x * mask for x in imgs]

        results.update(img=imgs)
        return results

@TRANSFORMS.register_module()
class ModalMask3D(BaseTransform):

    def __init__(self, img_drop_ratio: float = 0.0, 
                        each_img_drop_ratio: float = 0.0,
                        points_drop_ratio: float = 0.0) -> None:
        super(ModalMask3D, self).__init__()
        self.img_drop_ratio = img_drop_ratio
        self.each_img_drop_ratio = img_drop_ratio
        self.points_drop_ratio = points_drop_ratio

    def transform(self, input_dict: dict) -> dict:
        if np.random.rand() < self.img_drop_ratio:
            for i in range(len(input_dict['img'])):
                if np.random.rand() < self.each_img_drop_ratio:
                    input_dict['img'][i] = 0. * input_dict['img'][i]
            # input_dict['img'] = [0. * item for item in input_dict['img']]
        elif np.random.rand() < self.points_drop_ratio:
            input_dict['points'].tensor = input_dict['points'].tensor * 0.0

        # if self.mode == 'test':
        #     if self.mask_modal == 'image':
        #         input_dict['img'] = [0. * item for item in input_dict['img']]
        #     if self.mask_modal == 'points':
        #         input_dict['points'].tensor = input_dict['points'].tensor * 0.0
        # else:
        #     seed = np.random.rand()
        #     if seed > 0.75:
        #         input_dict['img'] = [0. * item for item in input_dict['img']]
        #     elif seed > 0.5:
        #         input_dict['points'].tensor = input_dict['points'].tensor * 0.0

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str
    

@TRANSFORMS.register_module()
class UnifiedObjectSample(BaseTransform):
    """Sample GT objects to the data.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(self, db_sampler, sample_2d=False, sample_method='depth', modify_points=False, mixup_rate=-1):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        self.sample_method = sample_method
        self.modify_points = modify_points
        self.mixup_rate = mixup_rate
        # print("db_sampler",db_sampler)
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler'
        # self.db_sampler = build_from_cfg(db_sampler, TRANSFORMS)
        self.db_sampler = TRANSFORMS.build(db_sampler)
        self.disabled = False

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def transform(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """
        if self.disabled:
            return input_dict
        
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']

        # change to float for blending operation
        points = input_dict['points']

        if 'gt_seg_map' in input_dict:
            gt_seg_map = input_dict['gt_seg_map']

        if self.sample_2d:
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                with_img=True)
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, with_img=False)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_points_idx = sampled_dict["points_idx"]
            sampled_gt_labels = sampled_dict['gt_labels_3d']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate(
                    [gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            points_idx = -1 * np.ones(len(points), dtype=np.int32)
            # check the points dimension
            # points = points.cat([sampled_points, points])
            points = points.cat([points, sampled_points])
            points_idx = np.concatenate([points_idx, sampled_points_idx], axis=0)

            #bev augmentation
            if 'gt_seg_map' in input_dict:
                class_det2seg = {
                    0: 4,
                    1: 5,
                    2: 7,
                    3: 6,
                    4: 8,
                    5: 10,
                    6: 2,
                    7: 3,
                    8: 0,
                    9: 1,
                }
                for i in range(len(sampled_gt_bboxes_3d)):
                    bev_corners = self.get_bbox_corners(sampled_gt_bboxes_3d[i])
                    label = class_det2seg[sampled_gt_labels[i]]
                    gt_seg_map = cv2.fillPoly(gt_seg_map, [bev_corners], label)

            if self.sample_2d:
                imgs = input_dict['img']
                # lidar2img = input_dict['lidar2img']
                lidar2cam = input_dict['lidar2cam']
                cam2img = input_dict['cam2img']
                lidar2img = np.array([cam2img[i] @ lidar2cam[i] for i in range(len(cam2img))])
                sampled_img = sampled_dict['images']
                sampled_num = len(sampled_gt_bboxes_3d)
                imgs, points_keep = self.unified_sample(imgs, lidar2img, 
                                            points.tensor.numpy(), 
                                            points_idx, gt_bboxes_3d.corners.numpy(), 
                                            sampled_img, sampled_num)
                
                input_dict['img'] = imgs

                if self.modify_points:
                    points = points[points_keep]

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.int64)
        input_dict['points'] = points

        return input_dict

    def get_bbox_corners(self, bbox, resolution=0.2, bev_shape=(512, 512)):
        """
        3D 바운딩 박스를 BEV 이미지의 코너로 변환.
        
        Args:
            bbox (list): [x, y, z, w, l, h, theta].
            resolution (float): BEV 해상도 (1 픽셀당 실제 거리, m).
            bev_shape (tuple): BEV 이미지 크기 (H, W).
        
        Returns:
            np.ndarray: BEV 이미지에서의 코너 좌표 (4, 2).
        """
        x, y, z, w, l, h, theta = bbox[:7]
        theta = theta - np.pi/2 # theta는 x축과의 각도
        # 월드 좌표에서의 코너 계산
        corners = np.array([
            [-l / 2, -w / 2],
            [-l / 2,  w / 2],
            [ l / 2,  w / 2],
            [ l / 2, -w / 2],
        ])
        
        # 회전 적용
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        rotated_corners = corners @ rotation_matrix.T
        rotated_corners += np.array([x, y])  # 중심 좌표 추가

        # 월드 좌표를 BEV 이미지 좌표로 변환
        bev_corners = rotated_corners / resolution
        bev_corners[:, 0] = bev_shape[1] / 2 + bev_corners[:, 0]  # x 방향 중심 이동
        bev_corners[:, 1] = bev_shape[0] / 2 + bev_corners[:, 1]  # y 방향 중심 이동
        return bev_corners.astype(int)

    def unified_sample(self, imgs, lidar2img, points, points_idx, bboxes_3d, sampled_img, sampled_num):
        # for boxes
        bboxes_3d = np.concatenate([bboxes_3d, np.ones_like(bboxes_3d[..., :1])], -1)
        is_raw = np.ones(len(bboxes_3d))
        is_raw[-sampled_num:] = 0
        is_raw = is_raw.astype(bool)
        raw_num = len(is_raw)-sampled_num
        # for point cloud
        points_3d = points[:,:4].copy()
        points_3d[:,-1] = 1
        points_keep = np.ones(len(points_3d)).astype(np.bool_)
        new_imgs = imgs

        assert len(imgs)==len(lidar2img) and len(sampled_img)==sampled_num
        for _idx, (_img, _lidar2img) in enumerate(zip(imgs, lidar2img)):
            coord_img = bboxes_3d @ _lidar2img.T
            coord_img[...,:2] /= coord_img[...,2,None]
            depth = coord_img[...,2]
            img_mask = (depth > 0).all(axis=-1)
            img_count = img_mask.nonzero()[0]
            if img_mask.sum() == 0:
                continue
            depth = depth.mean(1)[img_mask]
            coord_img = coord_img[...,:2][img_mask]
            minxy = np.min(coord_img, axis=-2)
            maxxy = np.max(coord_img, axis=-2)
            bbox = np.concatenate([minxy, maxxy], axis=-1).astype(int)
            bbox[:,0::2] = np.clip(bbox[:,0::2], a_min=0, a_max=_img.shape[1]-1)
            bbox[:,1::2] = np.clip(bbox[:,1::2], a_min=0, a_max=_img.shape[0]-1)
            img_mask = ((bbox[:,2:]-bbox[:,:2]) > 1).all(axis=-1)
            if img_mask.sum() == 0:
                continue
            depth = depth[img_mask]
            if 'depth' in self.sample_method:
                paste_order = depth.argsort()
                paste_order = paste_order[::-1]
            else:
                paste_order = np.arange(len(depth), dtype=np.int64)
            img_count = img_count[img_mask][paste_order]
            bbox = bbox[img_mask][paste_order]

            paste_mask = -255 * np.ones(_img.shape[:2], dtype=np.int32)
            fg_mask = np.zeros(_img.shape[:2], dtype=np.int32)
            # first crop image from raw image
            raw_img = []
            for _count, _box in zip(img_count, bbox):
                if is_raw[_count]:
                    raw_img.append(_img[_box[1]:_box[3],_box[0]:_box[2]])

            # then stitch the crops to raw image
            for _count, _box in zip(img_count, bbox):
                if is_raw[_count]:
                    if self.mixup_rate < 0:
                        _img[_box[1]:_box[3],_box[0]:_box[2]] = raw_img.pop(0)
                    else:
                        _img[_box[1]:_box[3],_box[0]:_box[2]] = \
                            _img[_box[1]:_box[3],_box[0]:_box[2]] * (1 - self.mixup_rate) + raw_img.pop(0) * self.mixup_rate
                    fg_mask[_box[1]:_box[3],_box[0]:_box[2]] = 1
                else:
                    img_crop = sampled_img[_count-raw_num]
                    if len(img_crop)==0: continue
                    img_crop = cv2.resize(img_crop, tuple(_box[[2,3]]-_box[[0,1]]))
                    if self.mixup_rate < 0:
                        _img[_box[1]:_box[3],_box[0]:_box[2]] = img_crop
                    else:
                        _img[_box[1]:_box[3],_box[0]:_box[2]] = \
                            _img[_box[1]:_box[3],_box[0]:_box[2]] * (1 - self.mixup_rate) + img_crop * self.mixup_rate

                paste_mask[_box[1]:_box[3],_box[0]:_box[2]] = _count
            
            new_imgs[_idx] = _img

            # calculate modify mask
            if self.modify_points:
                points_img = points_3d @ _lidar2img.T
                points_img[:,:2] /= points_img[:,2,None]
                depth = points_img[:,2]
                img_mask = depth > 0
                if img_mask.sum() == 0:
                    continue
                img_mask = (points_img[:,0] > 0) & (points_img[:,0] < _img.shape[1]) & \
                           (points_img[:,1] > 0) & (points_img[:,1] < _img.shape[0]) & img_mask
                points_img = points_img[img_mask].astype(int)
                new_mask = paste_mask[points_img[:,1], points_img[:,0]]==(points_idx[img_mask]+raw_num)
                raw_fg = (fg_mask == 1) & (paste_mask >= 0) & (paste_mask < raw_num)
                raw_bg = (fg_mask == 0) & (paste_mask < 0)
                raw_mask = raw_fg[points_img[:,1], points_img[:,0]] | raw_bg[points_img[:,1], points_img[:,0]]
                keep_mask = new_mask | raw_mask
                points_keep[img_mask] = points_keep[img_mask] & keep_mask

        return new_imgs, points_keep

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str