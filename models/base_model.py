import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.checkpoints_path, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_lr(self):
        return self.optimizers[0].param_groups[0]['lr']

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self, type):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        # errors_ret = OrderedDict()
        # for name in self.loss_names:
        #     if isinstance(name, str):
        #         errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        # return errors_ret
        if type == 'str':
            current_loss = 'loss_realA_realB:{0:.5f} | loss_G_GAN:{1:.5f} | loss_G_L:{2:.5f} | loss_G_percetual:{3:.5f} | loss_G_face:{4:.5f}'.format(
                self.loss_realA_realB.item(),
                self.loss_G_GAN.item(),self.loss_G_L1.item(),self.loss_G_percetual.item(),self.loss_G_face.item())
        elif type == 'dict':
            # if isinstance(self.loss_G_pre_perceptual, float):
            #     loss_G_pre_perceptual = self.loss_G_pre_perceptual
            # else:
            #     loss_G_pre_perceptual = self.loss_G_pre_perceptual.item()
            current_loss = {
                # "loss_G_pre_content": self.loss_G_pre_content,
                # "loss_G_pre_perceptual":loss_G_pre_perceptual,
                'loss_realA_realB':self.loss_realA_realB.item(),
                "loss_G_GAN":self.loss_G_GAN.item(),
                "loss_G_L":self.loss_G_L1.item(),
                "loss_G_percetual":self.loss_G_percetual.item(),
                "loss_G_face":self.loss_G_face.item(),
            }
        else:
            print("the type must be stricted in 'str' or 'dict'!")
            raise KeyError
        return current_loss

    def save_networks(self, epoch, iter):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_%s_net_%s.pth' % (epoch, iter, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.state_dict(), save_path)
                    # net.cuda()
                else:
                    torch.save(net.state_dict(), save_path)

    def save_latest_networks(self):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = 'net%s_latest.pth' % (name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                # if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                #     torch.save(net.state_dict(), save_path)
                #     # net.cuda()
                # else:
                torch.save(net.state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_latest_pretrain(self, opt, device):
        for name in self.model_names[0:2]:
            if isinstance(name, str):
                load_filename = 'net%s_latest.pth' %name
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                state_dict = torch.load(load_path, map_location=device)
                print('loading the model from %s' % load_path)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)                
        self.print_networks(opt.verbose)

    def load_given_pretrain(self, opt, device):
        for name in self.model_names[0:2]:
            if name == "G":
                load_filename = opt.load_netG_checkpoint_path
            if name == "D":
                load_filename = opt.load_netD_checkpoint_path
            # if name == "NU":
            #     load_filename = opt.load_netNU_checkpoint_path
            # load_path = os.path.join(self.save_dir, load_filename)
            load_path = load_filename
            net = getattr(self, 'net' + name)
            state_dict = torch.load(load_path, map_location=device)
            print('loading the model from %s' % load_path)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            # patch InstanceNorm checkpoints prior to 0.4
            for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
            net.load_state_dict(state_dict)                
        self.print_networks(opt.verbose)

    # def load_networks(self, epoch, device):
    #     """Load all the networks from the disk.

    #     Parameters:
    #         epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
    #     """
    #     for name in self.model_names:
    #         if isinstance(name, str):
    #             load_filename = '%s_net_%s.pth' % (epoch, name)
    #             load_path = os.path.join(self.save_dir, load_filename)
    #             net = getattr(self, 'net' + name)
    #             if isinstance(net, torch.nn.DataParallel):
    #                 net = net.module
    #                 print("this is DDP model and will load model from %s" % load_path)
    #                 state_dict = torch.load(load_path, map_location=device)
    #             else:
    #                 state_dict = torch.load(load_path, map_location=self.device)
    #                 print('loading the model from %s' % load_path)
    #             if hasattr(state_dict, '_metadata'):
    #                 del state_dict._metadata

    #             # patch InstanceNorm checkpoints prior to 0.4
    #             for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
    #                 self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
    #             net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


# class contrast_loss(torch.nn.Module):
#     # TODO: use precomputed face index map to only supervise the fore part.
#     def __init__(self,rate=1) -> None:
#         super(contrast_loss,self).__init__()
#         self.temp1 = 100.
#         self.temp2 = 10.
#         self.criterion=torch.nn.MSELoss()
#         self.fake = torch.zeros((16,128,128)).to('cuda:0')
#         self.rate = rate

#     def forward(self, src, tgt):
#         # if isinstance(src, np.ndarray):
#         #     src = torch.from_numpy(src)
#         # if isinstance(tgt, np.ndarray):
#         #     tgt = torch.from_numpy(tgt)

#         self.consist_loss = self.criterion(src,tgt)*self.temp1
#         # print(self.consist_loss)
#         if self.temp2 > 0:
#             self.temp2-=self.rate
#             self.differ_loss = -torch.log(self.criterion(src,self.fake.expand(src.shape[0],-1,-1,-1)))*self.temp2
#             return self.differ_loss+self.consist_loss
#         else:
#             return self.consist_loss

    # def load_networks(self, epoch, device):
    #     """Load all the networks from the disk.

    #     Parameters:
    #         epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
    #     """
    #     for name in self.model_names:
    #         if isinstance(name, str):
    #             load_filename = '%s_net_%s.pth' % (epoch, name)
    #             load_path = os.path.join(self.save_dir, load_filename)
    #             net = getattr(self, 'net' + name)
    #             if isinstance(net, torch.nn.DataParallel):
    #                 net = net.module
    #                 print("this is DDP model and will load model from %s" % load_path)
    #                 state_dict = torch.load(load_path, map_location=device)
    #             else:
    #                 state_dict = torch.load(load_path, map_location=self.device)
    #                 print('loading the model from %s' % load_path)
    #             if hasattr(state_dict, '_metadata'):
    #                 del state_dict._metadata

    #             # patch InstanceNorm checkpoints prior to 0.4
    #             for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
    #                 self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
    #             net.load_state_dict(state_dict)