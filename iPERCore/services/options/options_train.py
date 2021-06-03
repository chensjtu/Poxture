# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
from .options_base import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self._parser.add_argument("--dataset_dirs", type=str, default=["/date/dataset_share/iper/iper"], help="file containing test ids")
        self._parser.add_argument('--debug',action='store_true', help='whether use full txt')  
        self._parser.add_argument('--num_workers',type=int, default=6,  help='use multi-process for data') 
        # self._parser.add_argument('--num_workers',type=int, default=6,  help='use multi-process for data') 
        self._parser.add_argument('--num_gpus',type=int, default=3,  help='gpus for train') 

        self.is_train = True
        # pifu options
        self._parser.add_argument('--schedule', type=int, nargs='+', default=[0, 1, 2, 3], help='Decrease learning rate at these epochs.')
        self._parser.add_argument('--gamma', type=float, default=0.5, help='LR is multiplied by gamma on schedule.')
        self._parser.add_argument('--mlp_dim_color', nargs='+', default=[256, 512, 256, 128, 64, 32, 16],
                             type=int, help='# of dimensions of color mlp')
        self._parser.add_argument('--no_residual', action='store_true', help='no skip connection in mlp')
        self._parser.add_argument('--color_loss_type', type=str, default='contrast', help='mse | l1 | contrast')
        self._parser.add_argument('--temp', type=float, default='0.5', help='hyper_param for contrast loss')
        self._parser.add_argument('--norm_color', type=str, default='instance',
                             help='instance normalization or batch normalization or group normalization')
        self._parser.add_argument('--use_tanh', default = True,
                             help='using tanh after last conv of image_filter network')
        self._parser.add_argument('--learning_rateNU', type=float, default=1e-3, help='pre learning rate')
        self._parser.add_argument('--learning_ratep', type=float, default=1e-3, help='pix learning rate')
        # pix2pix options
        self._parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        self._parser.add_argument('--n_epochs_decay', type=int, default=5, help='number of epochs to linearly decay learning rate to zero')
        self._parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self._parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        self._parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        self._parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        self._parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self._parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        # self._parser.add_argument('--gpu_ids', type=int, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self._parser.add_argument('--input_nc', type=int, default=16, help='# of input image channels: 3 for RGB and 1 for grayscale')
        self._parser.add_argument('--isTrain', type=bool, default=True, help='for train make True')
        self._parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        self._parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        self._parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        self._parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        self._parser.add_argument('--netG', type=str, default='unet_256', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        self._parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        self._parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        self._parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        self._parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        self._parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        self._parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        self._parser.add_argument('--preprocess', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        # self._parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        self._parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        self._parser.add_argument('--checkpoints_path', default='./checkpoints',help='checkpoints')
        self._parser.add_argument('--name', type=str, default='pint',help='name of the experiment. ')
        self._parser.add_argument('--resume_epoch', type=int, default=-1, help='epoch resuming the training')
        self._parser.add_argument('--continue_train', action='store_true',default=False, help='continue training: load the latest model')
        self._parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        self._parser.add_argument('--results_path', type=str, default='./results', help='path to save results ply')
        # self._parser.add_argument('--load_checkpoint_path', type=str, help='path to load checkpoints') #the pix2pix checkpoint is decided by the func setup()
        self._parser.add_argument('--load_netG_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        self._parser.add_argument('--load_netD_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        self._parser.add_argument('--load_netNU_checkpoint_path', type=str, default=None, help='path to save checkpoints')  
        self._parser.add_argument('--source_img_folder', type=str, default="/date/dataset_share/iper/iper/primitives/001/1/1/processed", help='which image folder be source') 
        self._parser.add_argument('--source_img_id', type=int, default="0", help='which image folder be source') 
        self._parser.add_argument('--for_desired_txt', type=str, default="", help='which image folder be source') 

        

    def parse(self):
        cfg = super().parse()
        checkpoints_dir = cfg.meta_data.checkpoints_dir
        cfg = self.set_and_check_load_epoch(cfg, checkpoints_dir)

        return cfg

    def set_and_check_load_epoch(self, cfg, checkpoints_dir):
        if os.path.exists(checkpoints_dir):
            if cfg.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(checkpoints_dir):
                    if file.startswith("net_epoch_"):
                        epoch_name = file.split("_")[2]
                        if epoch_name.isdigit():
                            load_epoch = max(load_epoch, int(epoch_name))
                cfg.load_epoch = load_epoch
            else:
                found = False
                for file in os.listdir(checkpoints_dir):
                    if file.startswith("net_epoch_"):
                        found = int(file.split("_")[2]) == cfg.load_epoch
                        if found: break
                assert found, f"Model for epoch {cfg.load_epoch} not found"
        else:
            assert cfg.load_epoch < 1, f"Model for epoch {cfg.load_epoch} not found"
            cfg.load_epoch = 0

        return cfg
