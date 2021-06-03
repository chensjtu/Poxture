import sys
sys.path.append('/media/Diske/projects/Pint/iPERCore')
import os
import time
import tqdm
import unittest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import numpy as np
from easydict import EasyDict
import logging

from iPERCore.tools.human_digitalizer.renders import SMPLRenderer
from iPERCore.tools.human_pose3d_estimators.spin.runner import SPINRunner
from iPERCore.tools.utils.visualizers.visdom_visualizer import VisdomVisualizer
from iPERCore.data.processed_video_dataset import ProcessedVideoDataset
# from iPERCore.services.options.options_base import BaseOptions
from iPERCore.services.options.options_train import TrainOptions
from iPERCore.tools.trainers.base import FlowCompositionForTrainer

from lib.model import *

# pix2pix
from models.pix2pix_model import Pix2PixModel



def main(opt):
    # init device
    device = torch.device("cuda:0")
    # data root
    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s/imgs' % (opt.checkpoints_path, opt.name), exist_ok=True)
    # init logging
    logging.basicConfig(level=logging.INFO,
        filename='{}/{}/train.log'.format(opt.checkpoints_path, opt.name),
        filemode='a',
        format='%(asctime)s - %(pathname)s[line:%(lineno)d]: %(message)s')
    # init vis
    visualizer = VisdomVisualizer(
        env="test_train",
        ip="http://localhost",  # need to be replaced.
        port=8097  # need to be replaced.
    )

    # model initial
    pre_model = Pint_Model(opt).to(device)
    pix_model = Pix2PixModel(opt)
    pix_model.setup(opt)

    # continue train
    # pre_model.load_state_dict(torch.load(opt.pre_model_path, map_location=device))
    if opt.continue_train:
        if opt.load_netC_checkpoint_path == None:
            model_path_C = '%s/%s/netC_latest.pth' % (opt.checkpoints_path, opt.name)
            pix_model.load_latest_pretrain(opt,device) # load pix_model pretrained.
        else:
            model_path_C = os.path.join(opt.checkpoints_path,opt.name,opt.load_netC_checkpoint_path)
            pix_model.load_given_pretrain(opt,device) # load pix_model pretrained.
        print('Resuming from ', model_path_C)
        pre_model.load_state_dict(torch.load(model_path_C, map_location=device))
    # optimizer
    optimizer_pre_model = torch.optim.Adam(pre_model.parameters(), lr=opt.learning_rateC)
    lr = opt.learning_rateC
    #dataset
    train_dataset = ProcessedVideoDataset(opt, is_for_train=True)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=opt.serial_batches,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=None)
    data_len = len(trainloader)
    print('the dataset size is:', data_len)
    # model initial
    for epoch in tqdm(range(opt.n_epochs+opt.n_epochs_decay)):
        for i_train_batch, train_batch in enumerate(tqdm(trainloader)):
            iter_start_time = time.time()
            '''
            src img is in 0, and tgt img is at 1.
            images: size: B 2 3 512 512, represents the source image 
            masks: size: B 1 512 512, represents the tgt mask, bg is 0. and the fg is 1.
            uv_imgs: size: B 2 3 512 512, represents the double face man, [-1,1]
            ref_uvs: size: B 3 512 512, represents the broken uv of tgt, [-1,1]
            Ts: size: B 2 512 512 2, T matrix
            head_bboxs: size: B 4, head_bbox for tgt
            uv_index: size: B 3 512 512, int with the [0,1], 0 is unvisble and 1 is visble
            '''
            images = train_batch["images"].to(device, non_blocking=True)
            masks = train_batch["masks"].to(device, non_blocking=True)
            uv_imgs = train_batch["uv_imgs"].to(device, non_blocking=True)
            ref_uvs = train_batch["ref_uvs"].to(device, non_blocking=True)
            Ts = train_batch["Ts"].to(device, non_blocking=True)
            head_bboxs = train_batch["head_bboxs"].to(device, non_blocking=True)
            uv_index = train_batch["uv_index"].to(device, non_blocking=True)
            # ref_uvs[uv_index==1]
            # visualizer.vis_named_img('gt', ref_uvs)
            # visualizer.vis_named_img('canqueuv',ref_uvs*uv_index)
            
            fix_image, complete_feat, loss1, loss2 = pre_model(uv_imgs[:, 0],ref_uvs[:, 1]) # complete_feat:B 16 512 512

            # warp feature
            HD_img = F.grid_sample(complete_feat, Ts[:,1])
            # check_img = F.grid_sample(ref_uvs[:,1], Ts[:,1])
            # check_fix_img = F.grid_sample(fix_image, Ts[:,1])
            # check_gt = masks[:,1].expand(-1,3,-1,-1)*images[:,1]
            # visualizer.vis_named_img('check_warp', check_img)
            # visualizer.vis_named_img('check_gt', check_gt)
            # visualizer.vis_named_img('check_fix_img', check_fix_img)

            # pix model: time: 0.8s
            pix_model.set_input_with_face(HD_img, masks[:,1].expand(-1,3,-1,-1)*images[:,1], head_bboxs[:,1])
            pix_model.optimize_parameters(optimizer_pre_model,loss1,loss2)

            if not (i_train_batch+epoch*data_len) % opt.Train.print_freq_s:
                logging.log(logging.INFO, "lr_pre:%.6f, lr_GAN:%.6f"%(lr, pix_model.get_current_lr()))
                logging.log(logging.INFO, pix_model.get_current_losses())
            # if not (i_train_batch+epoch*data_len) % opt.Train.save_latest_freq_s:
            #     torch.save(pre_model.state_dict(), '%s/%s/netC_epoch_%d_iter_%d.pth' % (opt.checkpoints_path, opt.name, epoch, i_train_batch))
            #     pix_model.save_networks(epoch, i_train_batch)
            # visualizer for the fix_image
            # if not (i_train_batch+epoch*data_len) % opt.Train.display_freq_s:
            #     visualizer.vis_named_img('epoch: {},iter: {}, fix_img'.format(epoch,i_train_batch), fix_image[0:1])
            #     visualizer.vis_named_img('epoch: {},iter: {}, fake_B'.format(epoch,i_train_batch), pix_model.get_current_visuals()['fake_B'][0:1])
            #     visualizer.vis_named_img('epoch: {},iter: {}, real_B'.format(epoch,i_train_batch), pix_model.get_current_visuals()['real_B'][0:1])
            #     visualizer.vis_named_img('epoch: {},iter: {}, real_A'.format(epoch,i_train_batch), pix_model.get_current_visuals()['real_A'][0:1, :3])
            break
        torch.save(pre_model.state_dict(), '%s/%s/netC_latest.pth' % (opt.checkpoints_path, opt.name))
        pix_model.save_latest_networks()


if __name__ == "__main__":
    opt = TrainOptions().parse()
    main(opt)












