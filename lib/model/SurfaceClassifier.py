import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, num_views=1, no_residual=True, last_op=None):
        super(SurfaceClassifier, self).__init__()

        self.filters = []
        self.num_views = num_views
        self.no_residual = no_residual
        filter_channels = filter_channels
        self.last_op = last_op

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature):
        '''

        :param feature: list of [BxC_inxHxW] tensors of image features
        :param xy: [Bx3xN] tensor of (x,y) coodinates in the image plane
        :return: [BxC_outxN] tensor of features extracted at the coordinates
        '''

        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)

            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(
                    -1, self.num_views, y.shape[1], y.shape[2]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, feature.shape[1], feature.shape[2]
                ).mean(dim=1)

        if self.last_op:
            y = self.last_op(y)

        return y


class neural_texture(nn.Module):
    def __init__(self, filter_channels, num_views=1, no_residual=True, last_op=None):
        super(neural_texture, self).__init__()

        self.filters = []
        self.num_views = num_views
        self.no_residual = no_residual
        filter_channels = filter_channels
        self.last_op = last_op

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature):
        '''

        :param feature: list of [BxC_inxHxW] tensors of image features
        :param xy: [Bx3xN] tensor of (x,y) coodinates in the image plane
        :return: [BxC_outxN] tensor of features extracted at the coordinates
        '''

        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)

            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(
                    -1, self.num_views, y.shape[1], y.shape[2]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, feature.shape[1], feature.shape[2]
                ).mean(dim=1)

        if self.last_op:
            y = self.last_op(y)

        return y


class conv1_1(nn.Module):
    def __init__(self, input_layers=256, output_layers=16, kernel_size=1, stride=1, padding=0, bias=True):
        super(conv1_1, self).__init__()
        self.model = nn.Conv2d(in_channels=input_layers, out_channels=output_layers, \
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sig(x)

        return x*2-1


# class contrast_loss(nn.Module):
#     # TODO: use precomputed face index map to only supervise the fore part.
#     def __init__(self,rate=1) -> None:
#         super(contrast_loss,self).__init__()
#         self.temp1 = 100.
#         self.temp2 = 10.
#         self.criterion=nn.MSELoss()
#         self.fake = torch.zeros((16,128,128)).to('cuda:0')
#         self.rate = rate

#     def forward(self, src, tgt):
#         if isinstance(src, np.ndarray):
#             src = torch.from_numpy(src)
#         if isinstance(tgt, np.ndarray):
#             tgt = torch.from_numpy(tgt)

#         self.consist_loss = self.criterion(src,tgt)*self.temp1
#         # print(self.consist_loss)
#         if self.temp2 > 0:
#             self.temp2-=self.rate
#             self.differ_loss = -torch.log(self.criterion(src,self.fake.expand(src.shape[0],-1,-1,-1)))*self.temp2
#             return self.differ_loss+self.consist_loss
#         else:
#             return self.consist_loss

class period_loss(nn.Module):
    def __init__(self,r1=10,r2=1,r3=0.1,r4=0.01) -> None:
        super(period_loss,self).__init__()
        self.weights = [r1, r2, r3, r4]
        # self.slice = [4,8,12,16]
        self.criterion = nn.MSELoss()

    def forward(self,x,y):
        if x.shape[1]==16:
            loss = 0.0
            for i in range(4):
                loss+=self.weights[i] * self.criterion(x[:,4*i:4*i+4], y[:,4*i:4*i+4])
            return loss/4

        if x.shape[1]==3:  
            return self.criterion(x, y)


if __name__=='__main__':
    a = torch.rand((4,16,128,128))*2-1
    b = torch.rand((4,16,128,128))*2-1
    c = torch.ones((4,16,128,128))
    d = torch.zeros((4,16,128,128))
    ppp = period_loss(1,1,1,1)
    f = ppp(c,d)
    print(f)