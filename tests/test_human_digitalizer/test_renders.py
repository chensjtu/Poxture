# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.
import sys
sys.path.append('/media/Diske/projects/Pint')
import unittest
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from easydict import EasyDict

from iPERCore.tools.human_digitalizer.renders import SMPLRenderer
from iPERCore.tools.human_pose3d_estimators.spin.runner import SPINRunner
from iPERCore.tools.utils.visualizers.visdom_visualizer import VisdomVisualizer

# define visualizer
visualizer = VisdomVisualizer(
    env="test_renders",
    ip="http://localhost",  # need to be replaced.
    port=8097  # need to be replaced.
)


def load_image(img_path, image_size):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = image / 255 * 2 - 1

    # (h, w, c) -> (c, h, w)
    image = np.transpose(image, (2, 0, 1))
    return image


class TestRender(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.IMAGE_SIZE = 512
        cls.device = torch.device("cuda:0")

        cfg = EasyDict({
            "ckpt_path": "./assets/checkpoints/pose3d/spin_ckpt.pth",
            "smpl_path": "./assets/checkpoints/pose3d/smpl_model.pkl"
        })
        cls.spin_runner = SPINRunner(cfg, device=cls.device)

    def test_01_SMPLRenderer(self):

        render = SMPLRenderer().to(self.device)

        src_paths = [
            "/media/Diske/Datasets/iper/primitives/001/2/1/processed/images/0002.jpg"
        ]

        tgt_paths = [
            "/media/Diske/Datasets/iper/primitives/001/2/1/processed/images/0052.jpg"
        ]

        # 1.1 load source images
        src_imgs = []
        for im_path in src_paths:
            img = load_image(im_path, self.IMAGE_SIZE)
            src_imgs.append(img)
        src_imgs = np.stack(src_imgs, axis=0)
        src_imgs = torch.tensor(src_imgs).float().to(self.device)

        # 1.2 load target images
        tgt_imgs = []
        for im_path in tgt_paths:
            img = load_image(im_path, self.IMAGE_SIZE)
            tgt_imgs.append(img)
        tgt_imgs = np.stack(tgt_imgs, axis=0)
        tgt_imgs = torch.tensor(tgt_imgs).float().to(self.device)

        # 2.1 estimates smpls of source (cams, pose, shape)
        src_hmr_imgs = F.interpolate(src_imgs, size=(224, 224), mode="bilinear", align_corners=True)
        src_thetas = self.spin_runner.model(src_hmr_imgs)
        src_infos = self.spin_runner.get_details(src_thetas)

        # 2.1 estimates smpls of target (cams, pose, shape)
        tgt_hmr_imgs = F.interpolate(tgt_imgs, size=(224, 224), mode="bilinear", align_corners=True)
        tgt_thetas = self.spin_runner.model(tgt_hmr_imgs)
        tgt_infos = self.spin_runner.get_details(tgt_thetas)

        # 3.1 render fim and wim of UV
        bs = src_imgs.shape[0]
        img2uvs_fim, img2uvs_wim = render.render_uv_fim_wim(bs)
        f_uvs2img = render.get_f_uvs2img(bs)

        # 3.2 render fim and wim of source images
        src_f2verts, src_fim, _ = render.render_fim_wim(cam=src_infos["cam"], vertices=src_infos["verts"], smpl_faces=True)
        src_f2verts = render.get_vis_f2pts(src_f2verts, src_fim)

        # 4. warp source images to UV image
        base_one_map = torch.ones(bs, 1, self.IMAGE_SIZE, self.IMAGE_SIZE,
                                  dtype=torch.float32, device=self.device)
        Tsrc2uv = render.cal_bc_transform(src_f2verts, img2uvs_fim, img2uvs_wim)
        src_warp_to_uv = F.grid_sample(src_imgs, Tsrc2uv)
        vis_warp_to_uv = F.grid_sample(base_one_map, Tsrc2uv)
        merge_uv = torch.sum(src_warp_to_uv, dim=0, keepdim=True) / (
            torch.sum(vis_warp_to_uv, dim=0, keepdim=True) + 1e-5)

        # 5.1 warp UV image to source images
        src_f2verts, src_fim, src_wim = render.render_fim_wim(cam=src_infos["cam"], vertices=src_infos["verts"],
                                                              smpl_faces=True)
        Tuv2src = render.cal_bc_transform(f_uvs2img, src_fim, src_wim)
        uv_warp_to_src = F.grid_sample(src_warp_to_uv, Tuv2src)

        _, tgt_fim, tgt_wim = render.render_fim_wim(cam=tgt_infos["cam"], vertices=tgt_infos["verts"], smpl_faces=True)
        Tuv2tgt = render.cal_bc_transform(f_uvs2img, tgt_fim, tgt_wim)
        uv_warp_to_tgt = F.grid_sample(src_warp_to_uv, Tuv2tgt)

        uv_render_to_tgt, _ = render.forward(tgt_infos["cam"], tgt_infos["verts"], merge_uv, dynamic=False,
                                             get_fim=False)

        # 6. warp source to target
        Tsrc2tgt = render.cal_bc_transform(src_f2verts, tgt_fim, tgt_wim)
        src_warp_to_tgt = F.grid_sample(src_imgs, Tsrc2tgt)

        # 7. visualization
        visualizer.vis_named_img("vis_warp_to_uv", vis_warp_to_uv)
        visualizer.vis_named_img("src_warp_to_uv", src_warp_to_uv)
        visualizer.vis_named_img("uv_warp_to_src", uv_warp_to_src)
        visualizer.vis_named_img("uv_warp_to_tgt", uv_warp_to_tgt)
        visualizer.vis_named_img("uv_render_to_tgt", uv_render_to_tgt)
        visualizer.vis_named_img("src_warp_to_tgt", src_warp_to_tgt)

        src_fim_img, _ = render.encode_fim(None, None, fim=src_fim, transpose=True)
        tgt_fim_img, _ = render.encode_fim(None, None, fim=tgt_fim, transpose=True)
        visualizer.vis_named_img("src_fim", src_fim_img)
        visualizer.vis_named_img("src_wim", tgt_fim_img)

        visualizer.vis_named_img("tgt_fim", src_wim, transpose=True, denormalize=False)
        visualizer.vis_named_img("tgt_wim", tgt_wim, transpose=True, denormalize=False)


if __name__ == '__main__':
    unittest.main()
