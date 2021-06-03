# this code is for calculate the VGGloss, which also called percetual loss.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Union
import math

class VGG19(nn.Module):
    """
    Sequential(
          (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (1): ReLU(inplace)
          (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (3): ReLU(inplace)
          (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (6): ReLU(inplace)
          (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (8): ReLU(inplace)
          (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (11): ReLU(inplace)
          (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (13): ReLU(inplace)
          (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (15): ReLU(inplace)
          (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (17): ReLU(inplace)
          (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (20): ReLU(inplace)
          (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (22): ReLU(inplace)
          (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (24): ReLU(inplace)
          (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (26): ReLU(inplace)
          (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (29): ReLU(inplace)
          xxxx(30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          xxxx(31): ReLU(inplace)
          xxxx(32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          xxxx(33): ReLU(inplace)
          xxxx(34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          xxxx(35): ReLU(inplace)
          xxxx(36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    """

    def __init__(self, ckpt_path: Union[str, bool] = "./extradata/assets/pretrains/vgg19-dcbb9e9d.pth",
                 requires_grad=False, before_relu=False):
        super(VGG19, self).__init__()

        if False:
            vgg_pretrained_features = models.vgg19(pretrained=False).features
            ckpt = torch.load(ckpt_path, map_location="cpu")
            vgg_pretrained_features.load_state_dict(ckpt, strict=False)
        else:
            vgg_pretrained_features = models.vgg19(pretrained=True).features

        # print(f"Loading vgg19 from {ckpt_path}...")

        if before_relu:
            slice_ids = [1, 6, 11, 20, 29]
        else:
            slice_ids = [2, 7, 12, 21, 30]

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(slice_ids[0]):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[0], slice_ids[1]):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[1], slice_ids[2]):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[2], slice_ids[3]):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[3], slice_ids[4]):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_out1 = self.slice1(X)
        h_out2 = self.slice2(h_out1)
        h_out3 = self.slice3(h_out2)
        h_out4 = self.slice4(h_out3)
        h_out5 = self.slice5(h_out4)
        out = [h_out1, h_out2, h_out3, h_out4, h_out5]
        return out

class VGG16(nn.Module):
    """
        Sequential(
          (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (1): ReLU(inplace=True)
          (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (3): ReLU(inplace=True)
          (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (6): ReLU(inplace=True)
          (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (8): ReLU(inplace=True)
          (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (11): ReLU(inplace=True)
          (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (13): ReLU(inplace=True)
          (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (15): ReLU(inplace=True)
          (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (18): ReLU(inplace=True)
          (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (20): ReLU(inplace=True)
          (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (22): ReLU(inplace=True)
          (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (25): ReLU(inplace=True)
          xxxx(26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          xxxx(27): ReLU(inplace=True)
          xxxx(28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          xxxx(29): ReLU(inplace=True)
          xxxx(30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
    """

    def __init__(self, ckpt_path=False, requires_grad=False, before_relu=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        print("loading vgg16 ...")

        if before_relu:
            slice_ids = [1, 6, 11, 18, 25]
        else:
            slice_ids = [2, 7, 12, 19, 26]

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(slice_ids[0]):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[0], slice_ids[1]):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[1], slice_ids[2]):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[2], slice_ids[3]):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[3], slice_ids[4]):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_out1 = self.slice1(X)
        h_out2 = self.slice2(h_out1)
        h_out3 = self.slice3(h_out2)
        h_out4 = self.slice4(h_out3)
        h_out5 = self.slice5(h_out4)
        out = [h_out1, h_out2, h_out3, h_out4, h_out5]
        return out

class VGG11(nn.Module):
    """
    Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

      (1): ReLU(inplace=True)
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

      (4): ReLU(inplace=True)
      (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

      (7): ReLU(inplace=True)
      (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (9): ReLU(inplace=True)
      (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (11): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

      (12): ReLU(inplace=True)
      (13): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (14): ReLU(inplace=True)
      (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (16): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

      (17): ReLU(inplace=True)
      (18): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      ###(19): ReLU(inplace=True)
      ###(20): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )

    """
    def __init__(self, ckpt_path=False, requires_grad=False, before_relu=False):
        super(VGG11, self).__init__()
        vgg_pretrained_features = models.vgg11(pretrained=True).features
        print("loading vgg11 ...")

        if before_relu:
            slice_ids = [1, 4, 7, 12, 17]
        else:
            slice_ids = [2, 5, 8, 13, 18]

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(slice_ids[0]):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[0], slice_ids[1]):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[1], slice_ids[2]):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[2], slice_ids[3]):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[3], slice_ids[4]):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_out1 = self.slice1(X)
        h_out2 = self.slice2(h_out1)
        h_out3 = self.slice3(h_out2)
        h_out4 = self.slice4(h_out3)
        h_out5 = self.slice5(h_out4)
        out = [h_out1, h_out2, h_out3, h_out4, h_out5]
        return out

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class Percetual_loss(nn.Module):
    def __init__(self, 
                 before_relu=False, slice_ids=(0, 1, 2, 3, 4), vgg_type="VGG19", 
                 ckpt_path=False, resize=False, style_loss=False):
        super(Percetual_loss, self).__init__()
        self.device = torch.device('cuda')
        if vgg_type == "VGG19":
            self.vgg = VGG19(ckpt_path=ckpt_path, before_relu=before_relu).to(self.device)
        elif vgg_type == "VGG16":
            self.vgg = VGG16(ckpt_path=ckpt_path, before_relu=before_relu).to(self.device)
        else:
            self.vgg = VGG11(ckpt_path=ckpt_path, before_relu=before_relu).to(self.device)
        self.criterion = nn.L1Loss()
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.slice_ids = slice_ids
        self.style_loss = style_loss
        self.resize = resize

    def forward(self, x, y):
        '''
        x_vgg[i]: fake img
        y_vgg[i].detach(): ground_truth
        '''
        if self.resize:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=True)
            y = F.interpolate(y, size=(224, 224), mode="bilinear", align_corners=True)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        style_loss = 0
        if self.style_loss:
            for i in self.slice_ids:
                style_loss += F.mse_loss(gram_matrix(x_vgg[i]), gram_matrix(y_vgg[i].detach()))

        for i in self.slice_ids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss+style_loss


class Style_loss(nn.Module):
    # not used.
    def __init__(self, target_feature):
        super(Style_loss, self).__init__()
        self.vgg = VGG19(ckpt_path=ckpt_path, before_relu=before_relu).to(self.device)
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


if __name__ == "__main__":
    # percetual_loss = Percetual_loss(ckpt_path='./extradata/assets/checkpoints/losses/vgg19-dcbb9e9d.pth')
    percetual_loss = Percetual_loss()
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11_bn', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg13', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg13_bn', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16_bn', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19_bn', pretrained=True)
    with torch.no_grad():
        for i in range(10):
            loss = percetual_loss(torch.zeros((4,3,256,256),device=0), torch.ones((4,3, 256, 256),device=0))  
            print(loss)
    print("lalala")
