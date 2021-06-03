import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys
sys.path.append('/media/Diske/projects/Pint')
import tensorboardX
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

# from lib.mesh_util import *
# from lib.sample_util import *
# from lib.train_util import *
# from lib.data import *
from lib.model import *

# pix2pix
# from models.pix2pix_model import Pix2PixModel


def main(opt):
    # init
    # torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda:0")
    # data root
    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s/imgs' % (opt.checkpoints_path, opt.name), exist_ok=True)
    # model initial
    logging.basicConfig(level=logging.INFO,
        filename='{}/{}/train.log'.format(opt.checkpoints_path, opt.name),
        filemode='a',
        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    visualizer = VisdomVisualizer(
        env="3-25-longepoch",
        ip="http://localhost",  # need to be replaced.
        port=8097  # need to be replaced.
    )
    # init tensorboard
    named_tuple = time.localtime()
    time_string = time.strftime("%m_%d_%Y_%H_%M", named_tuple)
    log_dir = os.path.join(opt.checkpoints_path, opt.name, time_string)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = tensorboardX.SummaryWriter(logdir=log_dir)
    pre_model = Pint_Model(opt).to(device)
    # pix_model = Pix2PixModel(opt)
    # pix_model.setup(opt, device)
    # flow_comp = FlowCompositionForTrainer(opt=opt).to(device)
    # flow_comp.eval()
    # render = SMPLRenderer().to(device)
    # f_uvs2img = render.get_f_uvs2img(1)
    # load pretrain
    # pre_model.load_state_dict(torch.load("checkpoints/premodel_3/netC_epoch4_iter0.pth", map_location=device))
    if opt.continue_train:
        if opt.resume_epoch < 0:
            model_path_C = '%s/%s/netC_latest' % (opt.checkpoints_path, opt.name)
        else:
            model_path_C = '%s/%s/netC_epoch_%d' % (opt.checkpoints_path, opt.name, str(20))

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
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=None)
    data_len = len(trainloader)
    print('the dataset size is:', data_len)

    # model initial
    for epoch in tqdm(range(100)):
        for i_train_batch, train_batch in enumerate(trainloader):
            # iter_start_time = time.time()

            '''
            src img is in 0, and tgt img is at 1.
            images: size: B 2 3 512 512, represents the source image 
            masks: size: B 1 512 512, represents the tgt mask, bg is 0. and the fg is 1.
            uv_imgs: size: B 2 3 512 512, represents the double face man, [-1,1]
            ref_uvs: size: B 2 3 512 512, represents the broken uv of tgt, [-1,1]
            Ts: size: B 2 512 512 2, T matrix
            head_bboxs: size: B 4, head_bbox for tgt
            uv_index: size: B 3 512 512, int with the [0,1], 0 is unvisble and 1 is visble
            '''
            # images = train_batch["images"].to(device, non_blocking=True)
            # masks = train_batch["masks"].to(device, non_blocking=True)
            uv_imgs = train_batch["uv_imgs"].to(device, non_blocking=True)
            ref_uvs = train_batch["ref_uvs"].to(device, non_blocking=True)
            # Ts = train_batch["Ts"].to(device, non_blocking=True)
            # head_bboxs = train_batch["head_bboxs"].to(device, non_blocking=True)
            uv_index = train_batch["uv_index"].to(device, non_blocking=True) # ref_uvs[uv_index==1]
            
            # visualizer.vis_named_img('gt', ref_uvs)
            # visualizer.vis_named_img('canqueuv',ref_uvs*uv_index)
      
            # loss1:0.13, loss2:0.0003
            # premodel: 0.3s
            # A_feat, _, loss1, loss2 = pre_model(uv_imgs[:, 0], uv_imgs[:, 1], ref_uvs[:,1], uv_index) # complete_feat:B 16 512 512
            A_feat, _, loss1, loss2 = pre_model(ref_uvs[:,0],ref_uvs[:,1], ref_uvs[:,1], uv_index) # complete_feat:B 16 512 512

            optimizer_pre_model.zero_grad()
            if epoch < 30:
                loss1.backward()
            else:
                (loss1+loss2).backward()
            optimizer_pre_model.step()
            
            writer.add_scalar('train/loss1', loss1.item(), i_train_batch+epoch*data_len)
            writer.add_scalar('train/loss2', loss2.item(), i_train_batch+epoch*data_len)

            if not i_train_batch % 2000:
                visualizer.vis_named_img('epoch: {},iter: {}, src_uv'.format(epoch, i_train_batch), uv_imgs[0,0:1])
                # visualizer.vis_named_img('epoch: {},iter: {}, comple_uv'.format(epoch, i_train_batch), uv_imgs[0,1:])
                visualizer.vis_named_img('epoch: {},iter: {}, fix_img'.format(epoch, i_train_batch), A_feat[0:1])
                visualizer.vis_named_img('epoch: {},iter: {}, ref_uv'.format(epoch, i_train_batch), ref_uvs[0:1,1])

            if not i_train_batch % 50:
                logging.log(logging.INFO, 'epoch: {} | iter: {} | lr: {} | loss1 {} | loss2 {}'.format(epoch, i_train_batch,optimizer_pre_model.param_groups[0]['lr'],loss1.item(), loss2.item()))
            # if not i_train_batch % 200 and i_train_batch != 0:
            #     torch.save(pre_model.state_dict(), '%s/%s/netC_epoch%d_iter%d.pth' % (opt.checkpoints_path, opt.name, epoch, i_train_batch))
        if not epoch % 30:
            for param_group in optimizer_pre_model.param_groups:
                param_group['lr'] = param_group['lr']*0.3
            torch.save(pre_model.state_dict(), '%s/%s/netC_epoch_%d_iter_%d.pth' % (opt.checkpoints_path, opt.name, epoch, i_train_batch))
        # lr = adjust_learning_rate(optimizer_pre_model, epoch, lr, opt.schedule, opt.gamma)


if __name__ == "__main__":
    opt = TrainOptions().parse()
    main(opt)












