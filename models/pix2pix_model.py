import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
# from .losses import Percetual_loss, Style_loss
# from .face_loss import FaceLoss
from iPERCore.models.networks.criterions import VGGLoss, FaceLoss, LSGANLoss, TVLoss, TemporalSmoothLoss
from lib.model.Models import NestedUNet

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['complete_feat','real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'NU']
        else:  # during test time, only load G
            self.model_names = ['G', 'NU']
        # define networks (both generator and discriminator)
        self.netNU = NestedUNet(in_ch=3, out_ch=16 if opt.input_nc == 16 else opt.input_nc-3)
        self.netNU.cuda()
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.load_size = opt.image_size
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).cuda()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterion_smoothL1 = torch.nn.SmoothL1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_NU = torch.optim.Adam(self.netNU.parameters(), lr=opt.learning_rateNU)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.learning_ratep, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.learning_ratep, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self._init_losses()

    def _init_losses(self):
        # define loss functions
        # self.uv_loss = nn.SmoothL1Loss()
        # self.mse = nn.MSELoss()
        self.Percetual_loss = VGGLoss(vgg_type=self.opt.Train.use_vgg,
                                   ckpt_path=self.opt.Train.vgg_loss_path, resize=True).cuda()
        self.Face_loss = FaceLoss(pretrained_path=self.opt.Train.face_loss_path,
                                     factor=self.opt.Train.face_factor).cuda()
        # self.Contrast_loss = contrast_loss(rate=self.opt.temp).cuda()

        # init losses G
        # self.loss_G_pre_content = 0.0
        # self.loss_G_pre_perceptual = 0.0
        self.loss_G_percetual = 0.0
        self.loss_G_face = 0.0
        self.log = None
        self.loss_G_uv_content = 0.0
        self.loss_realA_realB = 0.0

    def get_network_G_D(self):
        return self.netG, self.netD

    def set_input(self, source, target):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            source: A
            target: B

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = source if AtoB else target
        self.real_B = target if AtoB else source
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.image_paths = 'deleted, not used'

    def set_input_with_face(self, target, face_bbox, uv_A, Ts, G_tsf):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            source: A
            target: B

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        
        AtoB = self.opt.direction == 'AtoB'
        self.real_B = target
        self.face_bbox = torch.clamp(face_bbox, 0., 511.)
        self.uv_A = uv_A
        self.Ts = Ts
        self.G_tsf = G_tsf
        self.masked_real_B = target

    def set_input_with_face_mask(self, target, face_bbox, uv_A, Ts, G_tsf, masks):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
        source: A
        target: B

        The option 'direction' can be used to swap images in domain A and domain B.
        """

        AtoB = self.opt.direction == 'AtoB'
        self.real_B = target
        self.face_bbox = torch.clamp(face_bbox, 0., 511.)
        self.uv_A = uv_A
        self.Ts = Ts
        self.G_tsf = G_tsf
        self.masked_real_B = masks.expand(-1,3,-1,-1)*target

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.complete_feat = self.netNU(self.uv_A)
        if self.opt.input_nc == 16:
            self.real_A = F.grid_sample(self.complete_feat, self.Ts)
        else:
            self.real_A = torch.cat((F.grid_sample(self.complete_feat, self.Ts), self.G_tsf), axis=1)
        self.fake_B = self.netG(self.real_A)  # G(A)


    def backward_D(self):
        """
        Calculate GAN loss for the discriminator
        """
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB.detach())
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """
        Calculate GAN and L1 loss for the generator
        containing error netC loss Mse()
        percetual loss VGG16()
        styleloss GEM matrix()
        faceloss SENet()
        """
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # Third, cal the percetual loss and style loss, this 
        self.loss_G_percetual = self.Percetual_loss(self.fake_B, self.real_B)*10
        self.loss_G_face = self.Face_loss(self.fake_B, self.real_B, bbox1=self.face_bbox, bbox2=self.face_bbox)*50
        self.loss_realA_realB = self.criterionL1(self.real_A[:,:3], self.masked_real_B)* self.opt.lambda_L1
        # self.loss_G_uv_content = self.criterionL1()

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1*2  + self.loss_G_percetual*5 + self.loss_G_face + self.loss_realA_realB
        #  + self.loss_realA_realB (in without RGB, this part is disabled)
        self.loss_G.backward()

    def optimize_parameters(self):
        # self.pre_model_error = pre_model_error
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_NU.zero_grad()
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        self.optimizer_NU.step()

    def test(self,uv_A, Ts, G_tsf):
        '''
        for test, do not need face or other loss related. 
        UVA : UV input
        Ts : sample T
        G_tsf : tsf of smpl from source
        '''

        self.complete_feat = self.netNU(uv_A)
        if self.opt.input_nc == 16:
            self.real_A = F.grid_sample(self.complete_feat, Ts)
        else:
            self.real_A = torch.cat((F.grid_sample(self.complete_feat, Ts), G_tsf), axis=1)
        self.fake_B = self.netG(self.real_A)  # G(A)
        return self.fake_B