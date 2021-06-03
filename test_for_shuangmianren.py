import sys
sys.path.append('/home/yangchen/projects/Pint')
import os
import time
from tqdm import tqdm
# import unittest
import torch
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP 
import logging
import cv2
import numpy as np
# from easydict import EasyDict

from iPERCore.tools.human_digitalizer.renders import SMPLRenderer
# from iPERCore.tools.human_pose3d_estimators.spin.runner import SPINRunner
from iPERCore.tools.utils.visualizers.visdom_visualizer import VisdomVisualizer
from iPERCore.tools.utils.filesio.cv_utils import save_cv2_img,standard_vis
from iPERCore.data.processed_video_dataset import ProcessedVideoDataset_for_cross_imitation_shuangmian
from iPERCore.services.options.options_train import TrainOptions
from iPERCore.tools.trainers.base import FlowCompositionForTrainer
# from lib.model import *
# pix2pix
from models.pix2pix_model import Pix2PixModel

def convert_to_single_GPU(state_dict_path, model):
    pretrained_dict = torch.load(state_dict_path)
    pretrained_dict = {
        k.replace('module.',''): v for k, v in pretrained_dict.items()
    }
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # return model

# this is for the test period
def main(opt):
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)     
    # video = cv2.VideoWriter('{}/{}/video.mp4'.format(opt.results_path, opt.name), cv2.VideoWriter_fourcc(*'mp4v'), 24, (1024,512))
    # init vis
    # visualizer = VisdomVisualizer(
    #     env=opt.name,
    #     ip="http://localhost",  # need to be replaced.
    #     port=3090  # need to be replaced.
    # )
    device = torch.device("cuda")
    # bg = cv2.cvtColor(cv2.imread("assets/temp/bg.jpg"), cv2.COLOR_BGR2RGB)
    # bg = torch.from_numpy(bg/255.*2-1).float().permute(2,0,1).unsqueeze(0).to(device)
    pix_model = Pix2PixModel(opt)
    # pix_model.setup(opt)
    print("loading the pretrained net .......")
    convert_to_single_GPU(opt.load_netNU_checkpoint_path, pix_model.netNU)
    convert_to_single_GPU(opt.load_netG_checkpoint_path, pix_model.netG)
    print("load net is down!")
    print("this is for shuangmian!")

    flow_comp = FlowCompositionForTrainer(opt=opt).to(device)
    flow_comp.eval()
    render = SMPLRenderer().to(device)
    f_uvs2img = render.get_f_uvs2img(1)

    train_dataset = ProcessedVideoDataset_for_cross_imitation_shuangmian(opt, is_for_train=False)
    testloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=False, # not opt.serial_batches
        num_workers=opt.num_workers,
        pin_memory=True)
    print("this dataset's length is %s"%len(testloader))

    pix_model.netNU.eval()
    pix_model.netG.eval()
    with torch.no_grad():
        for index, train_batch in enumerate(tqdm(testloader)):
            '''
            images: [B,2,3,512,512]
            masks: [B,2,1,512,512]
            uv: [B,3,512,512]
            smpls: [b,2,85]
            '''

            images = train_batch["images"].to(device, non_blocking=True)
            masks = train_batch["masks"].to(device, non_blocking=True)
            uv_imgs = train_batch["uv_imgs"].to(device, non_blocking=True)
            # Ts = train_batch["Ts"].to(device, non_blocking=True)
            # G_tsf = train_batch["G_tsf"].to(device, non_blocking=True)
            smpls = train_batch["smpls"].to(device, non_blocking=True)
            # to fix the unpaired problem.
            src_smpl = smpls[:,0:1]
            ref_smpl = smpls[:,1:2]

            # _, _, input_G_tsf, _, _, _, tsf_mask, head_bbox, _, uv_img, ref_uv, ref_fim, ref_wim = \
            #     flow_comp.mk_data(images, smpls, masks)

            # Tuv2ref = render.cal_bc_transform(f_uvs2img.expand(head_bbox.shape[0],-1,-1,-1), ref_fim, ref_wim)
            # visualizer.vis_named_img("111", F.grid_sample(uv_imgs, Tuv2ref))

            smpls[:,1:2][:,:,0:3] = smpls[:,0:1][:,:,0:3]
            smpls[:,1:2][:,:,-10:] = smpls[:,0:1][:,:,-10:]

            _, _, input_G_tsf, _, _, _, tsf_mask, head_bbox, _, uv_img, ref_uv, ref_fim, ref_wim = \
                flow_comp.mk_data(images, smpls, masks)

            Tuv2ref = render.cal_bc_transform(f_uvs2img.expand(head_bbox.shape[0],-1,-1,-1), ref_fim, ref_wim)
            # visualizer.vis_named_img("lalala", F.grid_sample(uv_imgs, Tuv2ref))

            fake = pix_model.test(uv_imgs, Tuv2ref, torch.flip(input_G_tsf.squeeze(1)[:,3:], dims=[1]))
            '''
            use whilt as bg and gray need fix
            # fake_with_bg = torch.where(masks.expand(-1,3,-1,-1)<0.5, bg, fake)
            # fake_in_one = torch.sum(torch.abs(fake), dim=1)
            # fake_mask = torch.sign(torch.clamp(fake_in_one-(1e-2), 0, 1))
            # fake_with_bg = torch.where(fake_mask>0.5, fake, torch.tensor(1.).to(device))
            # final_image = torch.cat((images, fake_with_bg), 3)[0].cpu().numpy()
            '''
            # new_fake = torch.where(masks[:,1].expand(-1,3,-1,-1) < 0.5, fake, torch.tensor(1.).to(device))
            # new_img = torch.where(masks[:,1].expand(-1,3,-1,-1) < 0.5, images[:,1], torch.tensor(1.).to(device))
            # final_image = torch.cat((new_img, fake), 3)[0].cpu().numpy()

            new_img = torch.where(masks[:,1].expand(-1,3,-1,-1) < 0.5, images[:,1], torch.tensor(1.).to(device))
            final_image = torch.cat((new_img, fake), 3)[0].cpu().numpy()

            # visualizer.vis_named_img("neural texture", standard_vis(pix_model.complete_feat[:,:3]))
            img = save_cv2_img(final_image, '{}/{}/{:05d}.jpg'.format(opt.results_path, opt.name, index),normalize=True)
        #     video.write(img)
        #     # if index > 119:
        #     #     break
        # video.release()
            # final_cv_img = cv2.cvtColor(((final_image[0].permute(1,2,0).cpu().numpy()+1)/2*255).astype(np.uint8)
            #     , cv2.COLOR_RGB2BGR)
            # cv2.imwrite('{}/{}/{:05d}.jpg'.format(opt.results_path, opt.name, index), final_cv_img)

    print("test is down!")


if __name__ == "__main__":
    opt = TrainOptions().parse()
    opt.is_train = False
    main(opt)



    # visualizer.vis_named_img("img", images)
    # visualizer.vis_named_img("img1",masks.expand(-1,3,-1,-1)*images)
    # visualizer.vis_named_img("img2", fake_with_bg)
    # visualizer.vis_named_img("img3", fake)
    # visualizer.vis_named_img("uV", uv_imgs)
    # visualizer.vis_named_img("tsf", G_tsf)
    # visualizer.vis_named_img("feat", standard_vis(pix_model.complete_feat[:, :3]))
    # visualizer.vis_named_img("realA", standard_vis(pix_model.real_A[:,:3]))
    # visualizer.vis_named_img("featqian1", standard_vis(pix_model.complete_feat[:, :1]))
    # visualizer.vis_named_img("featqian2", standard_vis(pix_model.complete_feat[:, 1:2]))
    # visualizer.vis_named_img("featqian3", standard_vis(pix_model.complete_feat[:, 2:3]))
    # visualizer.vis_named_img("feathou1", standard_vis(pix_model.complete_feat[:, 15:16]))
    # visualizer.vis_named_img("feathou2", standard_vis(pix_model.complete_feat[:, 14:15]))
    # visualizer.vis_named_img("feathou3", standard_vis(pix_model.complete_feat[:, 13:14]))
    # visualizer.vis_named_img("realA_qian1", standard_vis(pix_model.real_A[:, 0:1]))
    # visualizer.vis_named_img("realA_qian2", standard_vis(pix_model.real_A[:, 1:2]))
    # visualizer.vis_named_img("realA_qian3", standard_vis(pix_model.real_A[:, 2:3]))
    # visualizer.vis_named_img("realA_hou1", standard_vis(pix_model.real_A[:, 15:16]))
    # visualizer.vis_named_img("realA_hou2", standard_vis(pix_model.real_A[:, 14:15]))
    # visualizer.vis_named_img("realA_hou3", standard_vis(pix_model.real_A[:, 13:14]))

