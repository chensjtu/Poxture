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

# from lib.mesh_util import *
# from lib.sample_util import *
# from lib.train_util import *
# from lib.data import *
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
        format='%(asctime)s-[line:%(lineno)d]-: %(message)s')
    # init vis
    visualizer = VisdomVisualizer(
        env="visualizer",
        ip="http://localhost",  # need to be replaced.
        port=8097  # need to be replaced.
    )

    # model initial
    pre_model = Pint_Model(opt).to(device)
    pix_model = Pix2PixModel(opt)
    pix_model.setup(opt, device)
    # flow_comp = FlowCompositionForTrainer(opt=opt).to(device)
    # flow_comp.eval()
    # render = SMPLRenderer().to(device)
    # f_uvs2img = render.get_f_uvs2img(1)
    # load pretrain
    pre_model.load_state_dict(torch.load(opt.pre_model_path, map_location=device))
    if opt.continue_train:
        pix_model.load_pretrain(opt,device) # load pix_model pretrained.
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
        shuffle=opt.serial_batches,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=None)
    data_len = len(trainloader)
    print('the dataset size is:', data_len)
    # model initial
    for epoch in tqdm(range(opt.n_epochs+opt.n_epochs_decay)):
        for i_train_batch, train_batch in enumerate(trainloader):
            # iter_start_time = time.time()
            with torch.no_grad():

                images = train_batch["images"].to(device, non_blocking=True)
                # aug_bg = train_batch["bg"].to(device, non_blocking=True)
                smpls = train_batch["smpls"].to(device, non_blocking=True)
                masks = train_batch["masks"].to(device, non_blocking=True)
                # offsets = train_batch["offsets"].to(device, non_blocking=True) if "offsets" in train_batch else 0
                # links_ids = train_batch["links_ids"].to(device, non_blocking=True) if "links_ids" in inputs else None
                # opt.num_source
                # src_img = images[:, 0:opt.num_source].contiguous()
                # src_smpl = smpls[:, 0:opt.num_source].contiguous()
                # tsf_img = images[:, opt.num_source:].contiguous()
                # tsf_smpl = smpls[:, opt.num_source:].contiguous()
                # src_mask = masks[:, 0:opt.num_source].contiguous()
                # ref_mask = masks[:, opt.num_source:].contiguous()

                # input_G_bg, input_G_src, input_G_tsf, Tst, Ttt, src_mask, tsf_mask, head_bbox, body_bbox, uv_img, ref_uv, ref_fim, ref_wim = \
                # flow_comp(src_img, tsf_img, src_smpl, tsf_smpl, src_mask=src_mask, ref_mask=ref_mask)
                # conduct flow composition # TODO: tsf_mask is delated, which may cause difference
                # _, _, input_G_tsf, _, _, _, tsf_mask, head_bbox, _, uv_img, ref_uv, ref_fim, ref_wim = \
                #     flow_comp(src_img, tsf_img, src_smpl, tsf_smpl, src_mask=src_mask, ref_mask=ref_mask)
                # visualizer.vis_named_img("mask", tsf_mask)
                # visualizer.vis_named_img('tsf_img_masked', (1-tsf_mask).expand(-1,3,-1,-1)*tsf_img.squeeze(1))
                # visualizer.vis_named_img('uv_img', uv_img)
                # visualizer.vis_named_img('ref_uv', ref_uv.squeeze(1))
                '''
                input_G_tsf: B,nt,6,512,512
                tsf_mask: B,nt,1,512,512
                head_bbox: B ,4
                uv: B ,3 ,512 ,512
                ref_uv: B ,nt ,3 ,512 ,512
                ref_fim: B ,512 ,512
                ref_wim: B ,512 ,512 ,3
                '''
            fix_image,complete_feat, loss1, loss2  = pre_model(uv_img,ref_uv) # complete_feat:B 16 512 512
            # visualizer for the fix_image
            if not (i_train_batch+epoch*data_len) % opt.Train.display_freq_s:
                visualizer.vis_named_img('checkpoint: {},epoch: {},iter: {}, fix_img'.format(opt.pre_model_path,epoch, i_train_batch), fix_image)
            Tuv2src = render.cal_bc_transform(f_uvs2img.expand(head_bbox.shape[0],-1,-1,-1), ref_fim, ref_wim)
            HD_img = F.grid_sample(complete_feat, Tuv2src)
            HD_img_full = torch.cat((HD_img, input_G_tsf.squeeze(1)[:,3:]),dim=1)
            pix_model.set_input_with_face(HD_img_full, tsf_img.squeeze(1), head_bbox)
            pix_model.optimize_parameters(optimizer_pre_model,A_vis_feat, B_vis_uv,complete_feat, complete_feat_B)

            if not (i_train_batch+epoch*data_len) % opt.Train.print_freq_s:
                logging.log(logging.INFO, 'epoch: {},iter: {},lr0: {},lr1: {}'.format(
                    epoch, i_train_batch,optimizer_pre_model.param_groups[0]['lr'],pix_model.get_current_losses()))
                logging.log(logging.INFO, 'loss_G_pre_train:{} loss_G_GAN:{} loss_G_L1 :{} loss_G_percetual:{} loss_G_face:{}'.format(
                    pix_model.loss_G_pre_train.item(),pix_model.loss_G_GAN.item(),pix_model.loss_G_L1.item(),pix_model.loss_G_percetual.item(),pix_model.loss_G_face.item()))

if __name__ == "__main__":
    opt = TrainOptions().parse()
    main(opt)












