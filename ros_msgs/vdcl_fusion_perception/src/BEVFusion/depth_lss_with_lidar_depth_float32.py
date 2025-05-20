# modify from https://github.com/mit-han-lab/bevfusion
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS
from .ops import bev_mean_pool as bev_pool


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor(
        [row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2]
                           for row in [xbound, ybound, zbound]])
    return dx, bx, nx


class BaseViewTransform(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        # self.dx = nn.Parameter(dx, requires_grad=False)
        # self.bx = nn.Parameter(bx, requires_grad=False)
        # self.nx = nn.Parameter(nx, requires_grad=False)
        # self.dx = torch.tensor(dx, requires_grad=False, device='cuda')
        # self.bx = torch.tensor(bx, requires_grad=False, device='cuda')
        # self.nx = torch.tensor(nx, requires_grad=False, device='cuda')
        self.dx = dx.requires_grad_(False).to('cuda')
        self.bx = bx.requires_grad_(False).to('cuda')
        self.nx = nx.requires_grad_(False).to('cuda')

        self.C = out_channels
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound,
                         dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW))
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW,
                           dtype=torch.float).view(1, 1, fW).expand(D, fH, fW))
        ys = (
            torch.linspace(0, iH - 1, fH,
                           dtype=torch.float).view(1, fH, 1).expand(D, fH, fW))

        frustum = torch.stack((xs, ys, ds), -1)
        frustum = torch.tensor(frustum, requires_grad=False, device='cuda')
        # return nn.Parameter(frustum, requires_grad=False)
        return frustum

    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        **kwargs,
    ):
        B, N, _ = camera2lidar_trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots.float()).view(B, N, 1, 1, 1, 3,
                                          3).matmul(points.unsqueeze(-1)))
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins.float()))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if 'extra_rots' in kwargs:
            extra_rots = kwargs['extra_rots']
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3,
                                3).repeat(1, N, 1, 1, 1, 1, 1).matmul(
                                    points.unsqueeze(-1)).squeeze(-1))
        if 'extra_trans' in kwargs:
            extra_trans = kwargs['extra_trans']
            points += extra_trans.view(B, 1, 1, 1, 1,
                                       3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) /
                      self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([
            torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
            for ix in range(B)
        ])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = ((geom_feats[:, 0] >= 0)
                & (geom_feats[:, 0] < self.nx[0])
                & (geom_feats[:, 1] >= 0)
                & (geom_feats[:, 1] < self.nx[1])
                & (geom_feats[:, 2] >= 0)
                & (geom_feats[:, 2] < self.nx[2]))
        x = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    def forward(
        self,
        img,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x


class BaseDepthTransform(BaseViewTransform):

    def forward(
        self,
        img,
        points,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        batch_size = len(points)
        depth = torch.zeros(batch_size, img.shape[1], 1,
                            *self.image_size).to(points[0].device)

        for b in range(batch_size):
            cur_coords = points[b][:, :3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug
            # print(cur_coords.shape)
            # print(cur_lidar_aug_matrix.shape)
            # print(cur_img_aug_matrix.shape)
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3].float()).matmul(
                cur_coords.transpose(1, 0))
            # lidar2image
            # if cur_lidar2image.dim() == 2:
            #     cur_lidar2image = cur_lidar2image.unsqueeze(0)
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            # if cur_img_aug_matrix.dim() == 2:
            #     cur_img_aug_matrix = cur_img_aug_matrix.unsqueeze(0)
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = ((cur_coords[..., 0] < self.image_size[0])
                      & (cur_coords[..., 0] >= 0)
                      & (cur_coords[..., 1] < self.image_size[1])
                      & (cur_coords[..., 1] >= 0))
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth = depth.to(masked_dist.dtype)
                depth[b, c, 0, masked_coords[:, 0],
                      masked_coords[:, 1]] = masked_dist

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        # if camera2lidar_trans.dim() == 2:
        #     camera2lidar_trans = camera2lidar_trans.unsqueeze(0)
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        # ##visualize
        # import matplotlib.pyplot as plt
        # import numpy as np
        # import open3d as o3d
        # fig, ax = plt.subplots(2, 3, figsize=(12, 6))
        # fig2, ax2 = plt.subplots(2, 3, figsize=(12, 6))
        # fig3, ax3 = plt.subplots(2, 3, figsize=(12, 6))
        # for i in range(6):
        #     geom_points = geom[0, i].cpu().numpy()
        #     D,H,W,d = geom_points.shape
        #     # print(geom_points.shape) #118,32,88
        #     h_index = np.arange(H).reshape(1,H,1,1).repeat(D, axis=0).repeat(W, axis=2).reshape(-1,1)
        #     w_index = np.arange(W).reshape(1,1,W,1).repeat(D, axis=0).repeat(H, axis=1).reshape(-1,1)
        #     d_index = np.arange(D).reshape(D,1,1,1).repeat(H, axis=1).repeat(W, axis=2).reshape(-1,1)
        #     geom_points = np.concatenate([geom_points.reshape(-1,3), h_index, w_index, d_index], axis=1)

        #     ax[i // 3, i % 3].scatter(geom_points[:, 0], geom_points[:, 1], c=geom_points[:, 3], 
        #                                 cmap = 'viridis', s=1)
        #     ax[i // 3, i % 3].set_xlim(-100, 100)
        #     ax[i // 3, i % 3].set_ylim(-100, 100)
        #     ax[i // 3, i % 3].set_aspect('equal')
        #     ax[i // 3, i % 3].set_title(f'Image_{i}')

        #     ax2[i // 3, i % 3].scatter(geom_points[:, 0], geom_points[:, 1], c=geom_points[:, 4],
        #                                 cmap = 'viridis', s=1)
        #     ax2[i // 3, i % 3].set_xlim(-100, 100)
        #     ax2[i // 3, i % 3].set_ylim(-100, 100)
        #     ax2[i // 3, i % 3].set_aspect('equal')
        #     ax2[i // 3, i % 3].set_title(f'Image_{i}')

        #     ax3[i // 3, i % 3].scatter(geom_points[:, 0], geom_points[:, 1], c=geom_points[:, 5],
        #                                 cmap = 'viridis', s=1)
        #     ax3[i // 3, i % 3].set_xlim(-100, 100)
        #     ax3[i // 3, i % 3].set_ylim(-100, 100)
        #     ax3[i // 3, i % 3].set_aspect('equal')
        #     ax3[i // 3, i % 3].set_title(f'Image_{i}')
                                       

        #     # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(geom_points[:,:3]))
        #     # o3d.visualization.draw_geometries([pcd])

        # sample_idx = metas[0]['sample_idx']
        # fig.savefig(f'test_data/geom{sample_idx}_h.png')
        # fig2.savefig(f'test_data/geom{sample_idx}_w.png')
        # fig3.savefig(f'test_data/geom{sample_idx}_d.png')
        # plt.close()

        # pe = self.positional_embedding(depth, img_aug_matrix @ cam_intrinsic)
        pe = None

        x, x_depth = self.get_cam_feats(img, depth, pe, metas)
        x = self.bev_pool(geom, x)

        if self.vis:
            x_depth = self.bev_pool(geom, x_depth)
            ##visualize
            # print("geom", geom.shape)
            # print("x", x.shape)
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            import os
            # print("depth", depth.shape)
            # img_path = 'test_data'
            # files = [os.path.join(img_path,file) for file in os.listdir(img_path) if 'img' in file]
            # recent_file = max(files, key=os.path.getctime)
            # img_num = recent_file.split('.')[0].split('_')[-1]
            img_num = metas[0]['sample_idx']

            fig, ax = plt.subplots(figsize=(12, 12))
            bev_img = x[0].clone().detach().cpu().abs().max(0)[0]
            # bev_img_max = bev_img.max()
            # bev_img = torch.clamp(bev_img,max=bev_img_max*0.01)
            bev_img += bev_img.mean()
            # im = ax.imshow(bev_img, cmap='viridis')
            im = ax.imshow(bev_img, cmap='viridis',norm=colors.LogNorm(vmin=bev_img[bev_img>0].min(), vmax=bev_img.max()))
            fig.suptitle(f'Feature BEV')
            # fig.colorbar(im, ax=ax,location='right',label='Feature')
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2.5%", pad=0.05)  # 크기 줄이기 (기존보다 작게)
            fig.colorbar(im, cax=cax, label='Feature')
            fig.savefig(f'test_data/feature_bev_img_{img_num}.png')
            plt.close()

            fig, ax = plt.subplots(figsize=(12, 12))
            bev_img = x_depth[0].clone().detach().cpu().abs().max(0)[0]
            # bev_img_max = bev_img.max()
            # bev_img = torch.clamp(bev_img,max=bev_img_max*0.01)
            bev_img += bev_img.mean()
            # im = ax.imshow(bev_img, cmap='viridis')
            im = ax.imshow(bev_img, cmap='viridis',norm=colors.LogNorm(vmin=bev_img[bev_img>0].min(), vmax=bev_img.max()))
            fig.suptitle(f'Depth BEV')
            # fig.colorbar(im, ax=ax,location='right',label='Feature')
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2.5%", pad=0.05)  # 크기 줄이기 (기존보다 작게)
            fig.colorbar(im, cax=cax, label='Feature')
            fig.savefig(f'test_data/depth_bev_img_{img_num}.png')
            plt.close()

        return x
    
    def positional_embedding(self, x, intrinsics):
        B, N, C, H, W = x.shape
        u, v = torch.meshgrid(torch.arange(W, device=x.device), torch.arange(H, device=x.device), indexing='xy')
        u, v = u.float(), v.float()
        uv11 = torch.stack([u, v, torch.ones_like(u), torch.ones_like(u)], dim=0).reshape(4, -1)

        inv_intrinsics = torch.inverse(intrinsics.float())
        xyz = inv_intrinsics @ uv11
        print(xyz.shape)
        
        return xyz[:,:,:1,:,:]

@MODELS.register_module()
class DepthLSSTransform_with_lidar_depth(BaseDepthTransform):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        """Compared with `LSSTransform`, `DepthLSSTransform` adds sparse depth
        information from lidar points into the inputs of the `depthnet`."""
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            # nn.Conv2d(32, 64, 5, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

        self.vis = False

    def get_cam_feats(self, x, d, pe, metas=None):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:]).to(dtype=next(self.dtransform.parameters()).dtype)
        # pe = pe.view(B * N, *pe.shape[2:])
        x = x.view(B * N, C, fH, fW)
        # d_imgs = d.clone().detach().cpu()
        img_features = x.clone().detach().cpu()

        # print("dshape",d.shape)
        # depth_from_lidar = self.downsample_depth(d, 8, self.D)
        depth = self.downsample_depth(d, 8, self.D)

        d_transform = self.dtransform(d)

        # d_transform_imgs = d.clone().detach().cpu()
        # print("d", d_transform.shape)
        # print("x", x.shape)
        x = torch.cat([d_transform, x], dim=1)
        x = self.depthnet(x)

        # depth = x[:, :self.D].softmax(dim=1)

        # depth_loss = self.wasserstein_loss(depth, depth_from_lidar)
        
        x = depth.unsqueeze(1) * x.unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)

        if self.vis:
            d_imgs = d.clone().detach().cpu()
            d_transform_imgs = d_transform.clone().detach().cpu()
            
            # visualize, save cam_feature
            import matplotlib.pyplot as plt
            import numpy as np
            import os
            img_num = metas[0]['sample_idx']

            fig, ax = plt.subplots(2, 3, figsize=(12, 6))
            for i in range(6):
                img_feature = img_features[i].max(0)[0]
                im = ax[i//3, i%3].imshow(img_feature, cmap='viridis')
                ax[i//3, i%3].set_title('Image '+str(i))
            fig.suptitle(f'Cam Feature')
            fig.colorbar(im, ax=ax,location='right',label='Feature')
            plt.savefig(f'test_data/img_cam_feature_{img_num}.png')
            plt.close()
            
            # visualize, save depth img
            import matplotlib.pyplot as plt
            import numpy as np
            import os
            # # print("depth", depth.shape)
            # img_path = 'test_data'
            # files = [os.path.join(img_path,file) for file in os.listdir(img_path) if 'img' in file]
            # recent_file = max(files, key=os.path.getctime)
            # img_num = recent_file.split('.')[0].split('_')[-1]
            img_num = metas[0]['sample_idx']

            fig, ax = plt.subplots(2, 3, figsize=(12, 6))
            for i in range(6):
                depth_img = depth[i].clone().detach().cpu()
                # depth_bin = torch.linspace(1, 60, self.D).view(-1, 1, 1)
                depth_bin = torch.linspace(1, 102.5, self.D).view(-1, 1, 1)
                depth_img = (depth_img * depth_bin).sum(0)
                im = ax[i//3, i%3].imshow(depth_img, cmap='viridis', vmin=0, vmax=50)
                ax[i//3, i%3].set_title('Image '+str(i))
            # fig.suptitle(f'Estimated Depth, Wasserstein Loss: {depth_loss.item()}')
            fig.suptitle(f'Estimated Depth')
            fig.colorbar(im, ax=ax,location='right',label='Depth')
            plt.savefig(f'test_data/depth_img_{img_num}.png')
            plt.close()

            fig, ax = plt.subplots(2, 3, figsize=(12, 6))
            print("d_img", d_imgs.shape)
            for i in range(6):
                d_img = d_imgs[i][0]
                im = ax[i//3, i%3].imshow(d_img, cmap='viridis', vmin=0, vmax=50)
                ax[i//3, i%3].set_title('Image '+str(i))
            fig.suptitle(f'Lidar Projection Depth')
            fig.colorbar(im, ax=ax,location='right',label='Depth')
            plt.savefig(f'test_data/d_img_{img_num}.png')
            plt.close()

            fig, ax = plt.subplots(2, 3, figsize=(12, 6))
            for i in range(6):
                dd_img = self.downsample_depth_with_average(d_imgs[i][0],8)
                im = ax[i//3, i%3].imshow(dd_img, cmap='viridis', vmin=0, vmax=50)
                ax[i//3, i%3].set_title('Image '+str(i))
            fig.suptitle(f'Downsampled Lidar Projection Depth')
            fig.colorbar(im, ax=ax,location='right',label='Depth')
            plt.savefig(f'test_data/dd_img_{img_num}.png')
            plt.close()

            fig, ax = plt.subplots(2, 3, figsize=(12, 6))
            print("dtransform_img", d_transform_imgs.shape)
            for i in range(6):
                dtransform_img = d_transform_imgs[i].max(0)[0]
                im = ax[i//3, i%3].imshow(dtransform_img, cmap='viridis', vmin=0, vmax=10)
                ax[i//3, i%3].set_title('Image '+str(i))
            fig.suptitle(f'Estimated Depth Features')
            fig.colorbar(im, ax=ax,location='right',label='Depth')
            plt.savefig(f'test_data/dtransform_img_{img_num}.png')
            plt.close()

            # x_depth = depth.clone().view(B, N, 1, self.D, fH, fW)
            # x_depth = x_depth.permute(0, 1, 3, 4, 5, 2)
            x_depth = torch.ones_like(x)
            x_depth /= torch.sum(x_depth, dim=3, keepdim=True)
        else:
            x_depth = None

        return x, x_depth

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        # print("lss_x", x.shape)
        x = self.downsample(x)
        return x
    
    def downsample_depth_with_average(self, depth_map, scale):
        """
        Downsample a depth map by averaging valid points while ignoring zeros.

        Args:
            depth_map (torch.Tensor): Input depth map of shape (H, W).
            scale (int): Downsampling factor.

        Returns:
            torch.Tensor: Downsampled depth map of shape (H/scale, W/scale).
        """
        # Add a batch and channel dimension for grid operations
        depth_map = depth_map.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
        
        # Downsample using average pooling (ignoring zeros)
        valid_mask = (depth_map > 0).float()  # Mask of valid (non-zero) points
        
        # Sum of valid values in each patch
        depth_sum = F.avg_pool2d(depth_map * valid_mask, kernel_size=scale, stride=scale, divisor_override=1)
        
        # Count of valid values in each patch
        valid_count = F.avg_pool2d(valid_mask, kernel_size=scale, stride=scale, divisor_override=1)
        
        # Avoid division by zero
        valid_count[valid_count == 0] = 1  # To prevent NaN
        
        # Compute the average depth
        downsampled = depth_sum / valid_count
        
        # Remove added dimensions
        return downsampled.squeeze(0).squeeze(0)
    
    def downsample_depth(self, depth_map, scale, D):
        """
        Downsample a depth map by averaging valid points while ignoring zeros.

        Args:
            depth_map (torch.Tensor): Input depth map of shape (B, C, H, W).
            scale (int): Downsampling factor.
            D (int): Number of depth bins.

        Returns:
            torch.Tensor: Downsampled depth map of shape (B, D, H/scale, W/scale).
        """
        B, C, H, W = depth_map.shape  # Extract batch size, channels, height, and width
        new_H, new_W = H // scale, W // scale

        # Flatten depth map and compute valid mask
        y, x = torch.meshgrid(
            torch.arange(H, device=depth_map.device),
            torch.arange(W, device=depth_map.device),
            indexing="ij"
        )
        y, x = y.flatten(), x.flatten()
        depth_map_flat = depth_map.view(B, C, -1)
        valid_mask = depth_map_flat > 0

        # Process valid depths
        valid_depths = depth_map_flat[valid_mask]
        bin_indices = ((valid_depths - 1) // 0.5).long().clamp(min=0, max=D-2)
        weights = (valid_depths - 1) % 0.5

        # Compute downsampled indices
        downsampled_y = (y // scale).expand(B, C, -1)[valid_mask]
        downsampled_x = (x // scale).expand(B, C, -1)[valid_mask]
        batch_indices = torch.arange(B, device=depth_map.device).repeat_interleave(
            valid_mask.sum(dim=(1, 2))
        )

        # Create D-dimensional tensor with correct shape (B, D, new_H, new_W)
        D_channels = torch.zeros(B, D, new_H, new_W, device=depth_map.device)

        # Add values to bins with weights
        D_channels.index_put_(
            (batch_indices, bin_indices, downsampled_y, downsampled_x),
            (1 - weights),
            accumulate=True
        )
        D_channels.index_put_(
            (batch_indices, bin_indices + 1, downsampled_y, downsampled_x),
            weights,
            accumulate=True
        )

        # Normalize the depth channels
        normalization = D_channels.sum(dim=1, keepdim=True).clamp(min=1e-8)
        D_channels /= normalization

        return D_channels

    def wasserstein_loss(self, target, pred):
        """
        Compute Wasserstein Distance along the D dimension for distributions with shape (B, D, H, W).
        
        Args:
            target (torch.Tensor): Ground truth distribution (B, D, H, W).
            pred (torch.Tensor): Predicted distribution (B, D, H, W).

        Returns:
            torch.Tensor: Wasserstein distance averaged over B, H, and W.
        """
        # Validate input dimensions
        assert target.shape == pred.shape, "target and pred must have the same shape"
        B, D, H, W = target.shape

        # flatten the D dimension
        target = target.permute(0, 2, 3, 1).reshape(-1, D)
        pred = pred.permute(0, 2, 3, 1).reshape(-1, D)

        # make CDF
        target_cdf = target.cumsum(dim=1)
        pred_cdf = pred.cumsum(dim=1)

        # filter out zero values
        target_cdf_filtered = target_cdf[target_cdf[:,-1] > 0]
        pred_cdf_filtered = pred_cdf[target_cdf[:,-1] > 0]

        # Compute transport cost
        wasserstein_dist = torch.mean(torch.abs(target_cdf_filtered - pred_cdf_filtered))

        return wasserstein_dist

