import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
# print('this cuda devices is:', torch.)
import sys
sys.path.append('/media/Diske/projects/Pint/iPERCore')
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
import pickle as pkl

from iPERCore.tools.human_digitalizer.renders import SMPLRenderer
from iPERCore.tools.human_pose3d_estimators.spin.runner import SPINRunner
from iPERCore.tools.utils.visualizers.visdom_visualizer import VisdomVisualizer
from iPERCore.data.processed_video_dataset import ProcessedVideoDataset, ProcessedVideoDataset_for_mkdata
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
    # init
    # torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda:0")
    # mseloss = torch.nn.MSELoss()
    # mseloss = torch.nn.SmoothL1Loss()
    # data root
    # os.makedirs(opt.checkpoints_path, exist_ok=True)
    # os.makedirs(opt.results_path, exist_ok=True)
    # os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    # os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)
    # os.makedirs('%s/%s/imgs' % (opt.checkpoints_path, opt.name), exist_ok=True)
    # model initial
    # logging.basicConfig(level=logging.INFO,
    #     filename='{}/{}/train.log'.format(opt.checkpoints_path, opt.name),
    #     filemode='a',
    #     format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    visualizer = VisdomVisualizer(
        env="mk_data",
        ip="http://localhost",  # need to be replaced.
        port=8097  # need to be replaced.
    )
    # pre_model = Pint_Model(opt).to(device)
    # pix_model = Pix2PixModel(opt)
    # pix_model.setup(opt, device)
    flow_comp = FlowCompositionForTrainer(opt=opt).to(device)
    flow_comp.eval()
    render = SMPLRenderer().to(device)
    f_uvs2img = render.get_f_uvs2img(1)
    # load pretrain
    # if opt.continue_train:
    #     if opt.resume_epoch < 0:
    #         model_path_C = '%s/%s/netC_latest' % (opt.checkpoints_path, opt.name)
    #     else:
    #         model_path_C = '%s/%s/netC_epoch_%d' % (opt.checkpoints_path, opt.name, str(20))

    #     print('Resuming from ', model_path_C)
    #     pre_model.load_state_dict(torch.load(model_path_C, map_location=device))
    # # optimizer
    # optimizer_pre_model = torch.optim.Adam(pre_model.parameters(), lr=opt.learning_rateC)
    # lr = opt.learning_rateC
    #dataset

    train_dataset = ProcessedVideoDataset_for_mkdata(opt, is_for_train=True)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        sampler=None)
    print('the dataset size is:', len(trainloader))

    # model initial

    for i_train_batch, train_batch in enumerate(tqdm(trainloader)):
        # iter_start_time = time.time()
        with torch.no_grad():
            images = train_batch["images"].to(device)
            # aug_bg = train_batch["bg"].to(device, non_blocking=True)
            smpls = train_batch["smpls"].to(device)
            masks = train_batch["masks"].to(device)
            img_dir = train_batch["img_dir"]
            image_name = train_batch["image_name"]
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

            _, _, input_G_tsf, _, _, _, tsf_mask, head_bbox, _, uv_img, ref_uv, ref_fim, ref_wim = \
                flow_comp.mk_data(images, smpls, masks)
            '''
            input_G_tsf: B,nt,6,512,512
            tsf_mask: B,nt,1,512,512
            head_bbox: B ,4
            uv: B ,3 ,512 ,512
            ref_uv: B ,nt ,3 ,512 ,512
            ref_fim: B ,512 ,512
            ref_wim: B ,512 ,512 ,3
            '''
            # visualizer.vis_named_img('ref_uv', ref_uv[0])
            # visualizer.vis_named_img('input_G_tsf0', input_G_tsf[0][:,:3])
            # visualizer.vis_named_img('input_G_tsf1', input_G_tsf[0][:,3:])
            # visualizer.vis_named_img('tsf_mask', tsf_mask[0])
            # visualizer.vis_named_img('uv_img', uv_img[:1])
            # Tuv2src = render.cal_bc_transform(f_uvs2img.expand(head_bbox.shape[0],-1,-1,-1), ref_fim, ref_wim)
            for i in range(len(img_dir)):
                # file_path = img_dir[i][:-6]+'extra_data/'
                # uv_img_path = img_dir[i][:-6]+'uv_img/'
                # ref_uv_path = img_dir[i][:-6]+'ref_uv/'
                # input_G_tsf_path = img_dir[i][:-6]+'input_G_tsf/'
                uv_index = img_dir[i][:-6]+'uv_indx/' # note that in this project:0 means bg and 1 means fg

                # os.makedirs(file_path, exist_ok=True)
                # os.makedirs(uv_img_path, exist_ok=True)
                # os.makedirs(ref_uv_path, exist_ok=True)
                # os.makedirs(input_G_tsf_path, exist_ok=True)
                os.makedirs(uv_index, exist_ok=True)
                index_mask = (torch.sign(torch.abs(ref_uv[i].squeeze())).permute(1,2,0)*255).int()
                cv2.imwrite(os.path.join(uv_index, image_name[i][:-4]+'.jpg'), index_mask.cpu().numpy())



                # pkl_file={
                #     # 'image_path':image_name[i], #'000.jpg'
                #     # 'input_G_tsf':input_G_tsf[i].squeeze()[3:].cpu().numpy(),# (6, 512, 512) [-1,1]
                #     # 'tsf_mask':tsf_mask[i].cpu().numpy(),# torch.Size([1, 512, 512]) [0,1] # bg is 1, fg is 0
                #     'head_bbox':head_bbox[i].cpu().numpy(), # torch.Size([4])
                #     # 'uv_img':uv_img[i].cpu().numpy(), # torch.Size([ 3, 512, 512]) [-1,1]
                #     # 'ref_uv':ref_uv[i].squeeze().cpu().numpy(), # torch.Size([ 3, 512, 512]) [-1,1]
                #     'Tuv2src':Tuv2src[i].cpu().numpy() # torch.Size([512, 512, 2]) [-2,-1,1]
                # }
                # visualizer.vis_named_img('src', images[0][1:])
                # visualizer.vis_named_img('tsf_mask', tsf_mask[i:i+1])
                # visualizer.vis_named_img('full_uv', uv_img[i:i+1])

                # visualizer.vis_named_img('canqueuv_RGB', ref_uv[i:i+1].squeeze(1))
                # visualizer.vis_named_img('canqueuv_index', torch.sign(torch.abs(ref_uv[i])))
                # index_mask = torch.sign(torch.abs(ref_uv[i]))
                # visualizer.vis_named_img('test', ref_uv[i]*index_mask)
                # ref_uv[i][index_mask==0]=-1
                # visualizer.vis_named_img('black_test', ref_uv[i])

                # with open(os.path.join(file_path, image_name[i][:-4]+'.pkl'), "wb") as fp:
                #     pkl.dump(pkl_file, fp, protocol=2)

                # cv2.imwrite(os.path.join(uv_img_path, image_name[i][:-4]+'.jpg'), 
                #     cv2.cvtColor(((uv_img[i].permute(1,2,0)+1)/2*255).cpu().numpy().astype(np.uint8),cv2.COLOR_RGB2BGR))
                # cv2.imwrite(os.path.join(ref_uv_path, image_name[i][:-4]+'.jpg'), 
                #     cv2.cvtColor(((ref_uv[i].squeeze().permute(1,2,0)+1)/2*255).cpu().numpy().astype(np.uint8),cv2.COLOR_RGB2BGR))
                # cv2.imwrite(os.path.join(input_G_tsf_path, image_name[i][:-4]+'.jpg'), 
                #     cv2.cvtColor(((input_G_tsf[i].squeeze()[3:].permute(1,2,0)+1)/2*255).cpu().numpy().astype(np.uint8),cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    opt = TrainOptions().parse()
    main(opt)












