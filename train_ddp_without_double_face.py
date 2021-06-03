import sys
sys.path.append('/home/yangchen/projects/Pint')
import os
import time
import tqdm
import unittest
import torch
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP 
import logging
# import cv2
# import numpy as np
# from easydict import EasyDict

# from iPERCore.tools.human_digitalizer.renders import SMPLRenderer
# from iPERCore.tools.human_pose3d_estimators.spin.runner import SPINRunner
from iPERCore.tools.utils.visualizers.visdom_visualizer import VisdomVisualizer
from iPERCore.tools.utils.filesio.cv_utils import save_cv2_img,standard_vis
from iPERCore.data.processed_video_dataset import ProcessedVideoDataset_for_brokentexture
from iPERCore.services.options.options_train import TrainOptions
# from iPERCore.tools.trainers.base import FlowCompositionForTrainer
from lib.model import *
# pix2pix
from models.pix2pix_model import Pix2PixModel



def main(gpu, opt):
    #############################dist
    local_rank = opt.local_rank*opt.num_gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=opt.num_gpus,
        rank=local_rank
    )
    torch.manual_seed(0)
    ################################
    device = torch.device("cuda:{}".format(gpu))
    torch.cuda.set_device(gpu)
    print("this program will use gpu:{}".format(gpu))

    if dist.get_rank() == 0:
        # data root
        os.makedirs(opt.checkpoints_path, exist_ok=True)
        os.makedirs(opt.results_path, exist_ok=True)
        os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
        # os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)
        os.makedirs('%s/%s/imgs' % (opt.checkpoints_path, opt.name), exist_ok=True)
        # init logging
        logging.basicConfig(level=logging.INFO,
            filename='{}/{}/train.log'.format(opt.checkpoints_path, opt.name),
            filemode='a',
            format='%(asctime)s - %(pathname)s[line:%(lineno)d]: %(message)s')

        # init vis
        visualizer = VisdomVisualizer(
            env=opt.name,
            ip="http://localhost",  # need to be replaced.
            port=3090  # need to be replaced.
        )

    # model initial
    # pre_model = Pint_Model(opt).cuda()
    # pre_model.load_state_dict(torch.load(opt.pre_model_path, map_location=device))
    pix_model = Pix2PixModel(opt)
    pix_model.setup(opt)

    # wrap the model
    pix_model.netG = DDP(pix_model.netG, device_ids=[gpu])
    pix_model.netD = DDP(pix_model.netD, device_ids=[gpu])
    pix_model.netNU = DDP(pix_model.netNU, device_ids=[gpu])

    dist.barrier()
    # continue train
    # if dist.get_rank() == 0:
        # pre_model.load_state_dict(torch.load(opt.pre_model_path, map_location=device))
    if opt.continue_train:
        if opt.load_netNU_checkpoint_path == None:
            model_path_NU = '%s/%s/netNU_latest.pth' % (opt.checkpoints_path, opt.name)
            state = torch.load(model_path_NU, map_location=device)
            # for key in list(state.keys()):
            #     tmp = key.split('.', 2)
            #     newkey = tmp[0] + '.' + tmp[2]
            #     state[newkey] = state.pop(key)
            pix_model.netNU.load_state_dict(state)
            pix_model.load_latest_pretrain(opt,device) # load pix_model pretrained.
        else:
            # model_path_C = os.path.join(opt.checkpoints_path,opt.name,opt.load_netC_checkpoint_path)
            model_path_NU = opt.load_netNU_checkpoint_path
            state = torch.load(model_path_NU, map_location=device)
            # for key in list(state.keys()):
            #     tmp = key.split('.', 2)
            #     newkey = tmp[0] + '.' + tmp[2]
            #     state[newkey] = state.pop(key)
            pix_model.netNU.load_state_dict(state)
            pix_model.load_given_pretrain(opt,device) # load pix_model pretrained.
        print('Resuming from ', model_path_NU)
        # pre_model.load_state_dict(torch.load(model_path_NU, map_location=device))
    # optimizer
    # optimizer_pre_model = torch.optim.Adam(pre_model.parameters(), lr=opt.learning_rateC)
    lr = opt.learning_rateNU
    #dataset
    train_dataset = ProcessedVideoDataset_for_brokentexture(opt, is_for_train=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=opt.num_gpus,
        rank=local_rank
    )

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=False, # not opt.serial_batches
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)
    data_len = len(trainloader)
    # print('the dataset size is:', data_len)
    # model initial
    for epoch in tqdm(range(1000)):
        train_sampler.set_epoch(epoch)
        if dist.get_rank() == 0:
            logging.log(logging.INFO, "new epoch: %s starts: lr_pre:%.6f | lr_GAN:%.6f"%(epoch, lr, pix_model.get_current_lr()))
            print("new epoch: %s starts: lr_pre:%.6f | lr_GAN:%.6f"%(epoch, lr, pix_model.get_current_lr()))
        for i_train_batch, train_batch in enumerate(trainloader):
            # iter_start_time = time.time()
            '''
            src img is in 0, and tgt img is at 1.
            images: size: B 3 512 512
            masks: size: B 1 512 512
            uv_imgs: size: B 3 512 512
            ref_uvs: size: B 3 512 512
            Ts: size: B 512 512 2
            head_bboxs: size: B 2 4
            '''
            images = train_batch["images"].to(device, non_blocking=True)
            masks = train_batch["masks"].to(device, non_blocking=True)
            # uv_imgs = train_batch["uv_imgs"].to(device, non_blocking=True)
            ref_uvs = train_batch["ref_uvs"].to(device, non_blocking=True)
            Ts = train_batch["Ts"].to(device, non_blocking=True)
            head_bboxs = train_batch["head_bboxs"].to(device, non_blocking=True)
            G_tsf = train_batch["G_tsf"].to(device, non_blocking=True)
            # uv_index = train_batch["uv_index"].to(device, non_blocking=True) # ref_uvs[uv_index==1]

            # visualizer.vis_named_img("0", images[0:1])
            # visualizer.vis_named_img("1", masks[0:1])
            # visualizer.vis_named_img("2", ref_uvs[0:1])
            # visualizer.vis_named_img("3", G_tsf[0:1])
            # visualizer.vis_named_img("4", F.grid_sample(ref_uvs, Ts)[0:1])
            # break
            # pix model: time: 0.002s
            pix_model.set_input_with_face(masks.expand(-1,3,-1,-1)*images, head_bboxs, ref_uvs, Ts, G_tsf)
            pix_model.optimize_parameters()

            if dist.get_rank() == 0:
                if not (i_train_batch+epoch*data_len) % opt.Train.print_freq_s:
                    # logging.log(logging.INFO, "lr_pre:%.6f, lr_GAN:%.6f"%(lr, pix_model.get_current_lr()))
                    logging.log(logging.INFO, 'epoch:%s, iter:%s '%(epoch, i_train_batch)+pix_model.get_current_losses(type='str'))
                    # print('epoch:%s, iter:%s '%(epoch, i_train_batch)+pix_model.get_current_losses(type='str'))
                    visualizer.plot_current_losses(epoch, float(i_train_batch)/data_len, pix_model.get_current_losses(type='dict'))
                if not (i_train_batch+epoch*data_len) % opt.Train.display_freq_s:
                    # visualizer.vis_named_img('epoch: {},iter: {}, src'.format(epoch,i_train_batch), images[0:1, 0])
                    # visualizer.vis_named_img('epoch: {},iter: {}, fix_img'.format(epoch,i_train_batch), complete_feat[0:1,:3])
                    # visualizer.vis_named_img('epoch: {},iter: {}, complete_feat'.format(epoch,i_train_batch),
                        # standard_vis(pix_model.get_current_visuals()['complete_feat'][0:1,:3]))
                    visualizer.vis_named_img('epoch: {},iter: {}, fake_B'.format(epoch,i_train_batch), 
                        standard_vis(pix_model.get_current_visuals()['fake_B'][0:1]))
                    visualizer.vis_named_img('epoch: {},iter: {}, real_B'.format(epoch,i_train_batch), 
                        standard_vis(pix_model.get_current_visuals()['real_B'][0:1]))
                    # visualizer.vis_named_img('epoch: {},iter: {}, real_A'.format(epoch,i_train_batch), pix_model.get_current_visuals()['real_A'][0:1, :3])
                    save_cv2_img(pix_model.get_current_visuals()['fake_B'][0].detach().cpu().numpy(),
                        '%s/%s/imgs/epoch%s_iter%s.jpg' % (opt.checkpoints_path, opt.name, epoch,i_train_batch), normalize=True)
                if not (i_train_batch+epoch*data_len) % opt.Train.save_latest_freq_s and i_train_batch != 0:
                    # torch.save(pre_model.state_dict(), '%s/%s/netNU_epoch_%d_iter_%d.pth' % (opt.checkpoints_path, opt.name, epoch, i_train_batch))
                    pix_model.save_networks(epoch, i_train_batch)

        if dist.get_rank() == 0:
            # torch.save(pre_model.state_dict(), '%s/%s/netNU_latest.pth' % (opt.checkpoints_path, opt.name))
            pix_model.save_latest_networks()

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print('this program will use %s gpus'%n_gpus)
    opt = TrainOptions().parse()
    opt.num_gpus = n_gpus
    os.environ['MASTER_ADDR'] = '127.0.0.4'
    os.environ['MASTER_PORT'] = '8884'
    mp.spawn(main, nprocs=opt.num_gpus, args=(opt,), join=True)


# warp feature
# HD_img = F.grid_sample(complete_feat, Ts) #B 16 312 512
# check_img = F.grid_sample(ref_uvs[:,1], Ts[:,1])
# check_fix_img = F.grid_sample(fix_image, Ts[:,1])
# check_gt = masks[:,1].expand(-1,3,-1,-1)*images[:,1]·
# visualizer.vis_named_img('check_warp', check_img)
# visualizer.vis_named_img('check_gt', check_gt)
# visualizer.vis_named_img('check_fix_img', check_fix_img)

# visualizer.vis_named_img("0", images[0:1])
# visualizer.vis_named_img("1", masks[0:1])
# visualizer.vis_named_img("2", uv_imgs[0:1])
# visualizer.vis_named_img("3", G_tsf[0:1])
# visualizer.vis_named_img("4", F.grid_sample(uv_imgs, Ts)[0:1])