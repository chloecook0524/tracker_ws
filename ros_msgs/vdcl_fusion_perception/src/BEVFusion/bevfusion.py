from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization
import time


@MODELS.register_module()
class BEVFusion(Base3DDetector):

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        **kwargs,
    ) -> None:
        if data_preprocessor is not None:
            voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
            super().__init__(
                data_preprocessor=data_preprocessor, init_cfg=init_cfg)

            self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
            self.pts_voxel_layer = Voxelization(**voxelize_cfg)
        else:
            super().__init__(init_cfg=init_cfg)
            
        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

        self.fusion_layer = MODELS.build(
            fusion_layer) if fusion_layer is not None else None

        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)

        self.bbox_head = MODELS.build(bbox_head)

        self.init_weights()

        # self.voxel_numbers = []
        self.vis = False
        self.print_time = False
        self.random_aug = True

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        st_time = time.time()
        x = self.img_backbone(x)
        if self.print_time:
            print("img_backbone time:", time.time()-st_time)
        st_time = time.time()
        x = self.img_neck(x)
        if self.print_time:
            print("img_neck time:", time.time()-st_time)

        if not isinstance(x, torch.Tensor):
            # for i in range(len(x)):
            #     print(i, x[i].size())
            # ([6,256,32,88],[6,256,16,44])
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        if self.random_aug and self.training:
            if torch.rand(1)<0.1:
                # make random camera feature to zero
                aug_cam_ind = torch.randint(0, N, (1,)).item()
                x = x.clone()
                x[:, aug_cam_ind] = 0
                # print("aug_cam_ind:", aug_cam_ind)

        st_time = time.time()
        # with torch.autocast(device_type='cuda', dtype=torch.float32):
        #     x = self.view_transform(
        #         x,
        #         points,
        #         lidar2image,
        #         camera_intrinsics,
        #         camera2lidar,
        #         img_aug_matrix,
        #         lidar_aug_matrix,
        #         img_metas,
        #     )
        x = self.view_transform(
            x,
            points,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        if self.print_time:
            print("view_transform time:", time.time()-st_time)
        return x

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        st_time = time.time()
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1
        if self.print_time:
            print("voxelize time:", time.time()-st_time)

        st_time = time.time()
        feats = feats.float()
        coords = coords.int()
        with torch.cuda.amp.autocast(enabled=False):
            self.pts_middle_encoder = self.pts_middle_encoder.float()
            x = self.pts_middle_encoder(feats, coords, batch_size)
        if self.print_time:
            print("pts_middle_encoder time:", time.time()-st_time)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()
        
        # self.voxel_numbers.append(feats.size(0))
        # print("voxel_number:", feats.size(0))
        # print("max_voxel_number:", max(self.voxel_numbers))
        # print("min_voxel_number:", min(self.voxel_numbers))
        # print("mean_voxel_number:", sum(self.voxel_numbers) / len(self.voxel_numbers))
        return feats, coords, sizes

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:Det3DDataSample]): The Data
                Samples. It usually includes information such as
                gt_instance_3d.

        Returns:
            list[:obj:Det3DDataSample]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the pred_instances_3d usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:BaseInstance3DBoxes): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """

        # # save batch_inputs_dict
        # import pickle
        # with open('batch_inputs_dict.pkl', 'wb') as f:
        #     pickle.dump(batch_inputs_dict, f)
        # # save batch_data_samples
        # with open('batch_data_samples.pkl', 'wb') as f:
        #     pickle.dump(batch_data_samples, f)
        # print("batch_inputs_dict:", batch_inputs_dict)
        # print("batch_data_samples:", batch_data_samples)
        # print("batch_data_samples:", len(batch_data_samples))

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        st_time = time.time()
        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)
        if self.print_time:
            print("bbox_head predict time:", time.time()-st_time)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)
        return res

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature = self.extract_img_feat(imgs, deepcopy(points),
                                                lidar2image, camera_intrinsics,
                                                camera2lidar, img_aug_matrix,
                                                lidar_aug_matrix,
                                                batch_input_metas)
            # features.append(img_feature)

            if self.vis:
                ## visualize img_feature
                import matplotlib.pyplot as plt
                print("img_feature:", img_feature.size())
                sample_idx = batch_input_metas[0]['sample_idx']
                # img_feature_vis = img_feature[0].clone().sum(dim=0).detach().cpu().numpy()
                img_feature_vis = img_feature[0].clone().detach().cpu().abs().max(axis=0).values.numpy()
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                im = ax.imshow(img_feature_vis, cmap='viridis')
                fig.colorbar(im, ax=ax)
                fig.suptitle("img_feature")
                plt.savefig(f"test_data/img_feature_{sample_idx}.png")
                plt.close()

                aug_img_feature = torch.zeros((3,)+img_feature.shape[1:]).cuda()
                aug_img_feature[0] = img_feature[0]
                aug_img_feature[2] = img_feature[0]
                features.append(aug_img_feature)
            else:
                features.append(img_feature)

        pts_feature = self.extract_pts_feat(batch_inputs_dict)
        if self.random_aug and self.training:
            if torch.rand(1)<0.05:
                pts_feature = torch.zeros_like(pts_feature).cuda()    
                # print("lidar feature is zero")            
            elif torch.rand(1)<0.2:
                pts_feature = pts_feature.clone()
                num_row = pts_feature.size(1)
                num_column = pts_feature.size(2)
                num_zero_points = int(num_row * num_column * 0.2)
                aug_pts_ind = torch.randint(0, num_row * num_column, (num_zero_points,))
                row = aug_pts_ind // num_column
                column = aug_pts_ind % num_column
                pts_feature[:, row, column] = 0
                # print("aug_pts_ind:", len(aug_pts_ind))
        # features.append(pts_feature)
        
        if self.vis:
            ## visualize pts_feature
            import matplotlib.pyplot as plt
            print("pts_feature:", pts_feature.size())
            sample_idx = batch_input_metas[0]['sample_idx']
            # pts_feature_vis = pts_feature[0].clone().sum(dim=0).detach().cpu().numpy()
            pts_feature_vis = pts_feature[0].clone().detach().cpu().abs().max(axis=0).values.numpy()
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            im = ax.imshow(pts_feature_vis, cmap='viridis')
            fig.colorbar(im, ax=ax)
            fig.suptitle("pts_feature")
            plt.savefig(f"test_data/pts_feature_{sample_idx}.png")
            plt.close()

            aug_pts_feature = torch.zeros((3,)+pts_feature.shape[1:]).cuda()
            aug_pts_feature[1] = pts_feature[0]
            aug_pts_feature[2] = pts_feature[0]
            features.append(aug_pts_feature)
        else:
            features.append(pts_feature)

        st_time = time.time()
        if self.fusion_layer is not None:
            x = self.fusion_layer(features)      
            if self.print_time:
                print("fusion_layer time:", time.time()-st_time)  
        else:
            assert len(features) == 1, features
            x = features[0]

        if self.vis:
            fused_feature = x[0].clone().detach().cpu().numpy()

        st_time = time.time()
        x = self.pts_backbone(x)
        if self.print_time:
            print("pts_backbone time:", time.time()-st_time)
        if self.vis:
            backboned_feature = x[0][2].clone().detach().cpu().numpy()

        
        st_time = time.time()
        x = self.pts_neck(x)
        if self.print_time:
            print("pts_neck time:", time.time()-st_time)
        if self.vis:
            necked_feature = x[0][2].clone().detach().cpu().numpy()

        if self.vis:
            ## visualize fused_feature
            import matplotlib.pyplot as plt
            print("fused_feature:", fused_feature.shape)
            sample_idx = batch_input_metas[0]['sample_idx']
            fused_feature_vis = np.abs(fused_feature).max(axis=0)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            im = ax.imshow(fused_feature_vis, cmap='viridis')
            fig.colorbar(im, ax=ax)
            fig.suptitle("fused_feature")
            plt.savefig(f"test_data/fused_feature_max_{sample_idx}.png")
            plt.close()
            fused_feature_vis = np.abs(fused_feature).mean(axis=0)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            im = ax.imshow(fused_feature_vis, cmap='viridis')
            fig.colorbar(im, ax=ax)
            fig.suptitle("fused_feature")
            plt.savefig(f"test_data/fused_feature_{sample_idx}.png")
            plt.close()


            ## visualize backboned_feature
            import matplotlib.pyplot as plt
            print("backboned_feature:", backboned_feature.shape)
            sample_idx = batch_input_metas[0]['sample_idx']
            backboned_feature_vis = np.abs(backboned_feature).max(axis=0)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            im = ax.imshow(backboned_feature_vis, cmap='viridis')
            fig.colorbar(im, ax=ax)
            fig.suptitle("backboned_feature")
            plt.savefig(f"test_data/backboned_feature_max_{sample_idx}.png")
            plt.close()
            backboned_feature_vis = np.abs(backboned_feature).mean(axis=0)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            im = ax.imshow(backboned_feature_vis, cmap='viridis')
            fig.colorbar(im, ax=ax)
            fig.suptitle("backboned_feature")
            plt.savefig(f"test_data/backboned_feature_{sample_idx}.png")
            plt.close() 

            ## visualize necked_feature
            import matplotlib.pyplot as plt
            print("necked_feature:", necked_feature.shape)
            sample_idx = batch_input_metas[0]['sample_idx']
            necked_feature_vis = np.abs(necked_feature).max(axis=0)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            im = ax.imshow(necked_feature_vis, cmap='viridis')
            fig.colorbar(im, ax=ax)
            fig.suptitle("necked_feature")
            plt.savefig(f"test_data/necked_feature_max_{sample_idx}.png")
            plt.close()
            necked_feature_vis = np.abs(necked_feature).mean(axis=0)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            im = ax.imshow(necked_feature_vis, cmap='viridis')
            fig.colorbar(im, ax=ax)
            fig.suptitle("necked_feature")
            plt.savefig(f"test_data/necked_feature_{sample_idx}.png")
            plt.close()

        return x

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.vis:
            # batch_data_samples = batch_data_samples.repeat_interleave(3,dim=0)
            batch_data_samples = [item for item in batch_data_samples for _ in range(3)]

        losses = dict()
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)

        losses.update(bbox_loss)

        return losses