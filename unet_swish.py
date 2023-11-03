# writed by Buyu
# Used to find particle centroid

import torch
import torch.nn as nn
import torch.nn.functional as F


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x
# def swish(x):
#     x = x * torch.sigmoid(x)
#     return x

def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

# class conv3(nn.Module):
#     def __init__(self, inputChannels, outputChannels):
#         self.inputChannels  =inputChannels
#         self.outputChannels = outputChannels
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(inputChannels, outputChannels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(outputChannels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x):
#         conv = self.conv(x)
#
#         return conv

def conv3(inputChannels, outputChannels):
    conv = nn.Sequential(
        nn.Conv2d(inputChannels, outputChannels, kernel_size=3,stride=1, padding=1),
        nn.BatchNorm2d(outputChannels,eps=1e-05, momentum=0.1, affine=True,track_running_stats=True),
        swish(),
        # nn.ReLU(inplace=True),
    )
    return conv

class resunit(nn.Module):
    def __init__(self, inputChannels, outputChannels, direction):
        super(resunit, self).__init__()

        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.direction = direction
        branchChannels = inputChannels // 2
        self.branch1 = nn.Sequential()

        # this version use 3x3 two times
        self.branch2 = nn.Sequential(
            nn.Conv2d(branchChannels, branchChannels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(branchChannels),
            swish(),
            # nn.ReLU(inplace=True)
            nn.Conv2d(branchChannels, branchChannels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(branchChannels),
            swish(),
            # nn.ReLU(inplace=True)

        )
        # this version use the same structure as Shufflenet did
        # self.branch2 = nn.Sequential(
        #     nn.BatchNorm2d(branchChannels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(branchChannels, branchChannels, kernel_size=(1, 1), stride=(1, 1), bias=False),
        #     nn.BatchNorm2d(branchChannels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(branchChannels, branchChannels,  stride=1, padding=0, bias=False, groups=branchChannels),
        #     nn.BatchNorm2d(branchChannels),
        #     nn.Conv2d(branchChannels, branchChannels, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(branchChannels),
        #     nn.ReLU(inplace=True),
        # )
        self.conv3 = conv3(self.inputChannels, self.outputChannels)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat((x1, self.branch2(x2)), dim=1)
        if self.direction == 0:
            out = channel_shuffle(out, 2)
        elif self.direction ==1:
            out = self.conv3(out)
        return out

def downPool(inputChannels):
    afterDownPool = nn.Sequential(
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(inputChannels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        swish(),
        # nn.ReLU(inplace=True)

    )
    return afterDownPool

def deconv(inputChannels):
    afterDeconv = nn.Sequential(
        nn.ConvTranspose2d(inputChannels, inputChannels, kernel_size=2, stride=2),
        nn.BatchNorm2d(inputChannels),
        swish(),
        # nn.ReLU(inplace=True)
    )
    return afterDeconv


def conv1(inputChannels,outputChannels):

    conv = nn.Sequential(
                nn.Conv2d(inputChannels, outputChannels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(outputChannels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                swish(),)
            # nn.ReLU(inplace=True)

    return conv

class uNet(nn.Module):
    def __init__(self, respart = resunit):
        super(uNet, self).__init__()

        # down pool part
        self.downPool_1 = downPool(64)
        self.downPool_2 = downPool(128)
        self.downPool_3 = downPool(256)
        # encoder conv
        self.conv3Down_1 = conv3(1,64)  #Designed for grayscale images
        self.conv3Down_2 = conv3(64,128)
        self.conv3Down_3 = conv3(128,256)
        self.conv3Down_4 = conv3(256,512)
        # decoder conv
        self.conv3Up_1 = conv3(256, 128)
        self.conv3Up_2 = conv3(128, 64)
        self.conv3Up_3 = conv3(64, 64)
        # up transfer
        self.deconv_1 = deconv(256)
        self.deconv_2 = deconv(128)
        self.deconv_3 = deconv(64)
        # change channels
        self.conv1 = conv1(64, 1)
        # encoder resunit
        self.resunitDown_1 = respart(64, 64,0)
        self.resunitDown_2 = respart(128,128 ,0)
        self.resunitDown_3 = respart(256,256 ,0)
        # decoder resunit
        self.resunitUp_1 = respart(512,256 ,1)
        self.resunitUp_2 = respart(512,256 ,1)
        self.resunitUp_3 = respart(256,128 ,1)
        self.resunitUp_4 = respart(128,64 ,1)

    def forward(self, x):
        # encoder part
        xEncoder0_1 = self.conv3Down_1(x)
        xEncoder0_2 = self.resunitDown_1(xEncoder0_1)
        xEncoder1_0 = self.downPool_1(xEncoder0_2)
        xEncoder1_1 = self.conv3Down_2(xEncoder1_0)
        xEncoder1_2 = self.resunitDown_2(xEncoder1_1)
        xEncoder2_0 = self.downPool_2(xEncoder1_2)
        xEncoder2_1 = self.conv3Down_3(xEncoder2_0)
        xEncoder2_2 = self.resunitDown_3(xEncoder2_1)
        xEncoder3_0 = self.downPool_3(xEncoder2_2)
        xEncoder3_1 = self.conv3Down_4(xEncoder3_0)

        # decoder part

        xDecoder0_0 = self.resunitUp_1(xEncoder3_1)
        xDecoder1_0 = self.deconv_1(xDecoder0_0)
        xDecoder1_1 = self.resunitUp_2(torch.cat([xEncoder2_2, xDecoder1_0],1))
        xDecoder1_2 = self.conv3Up_1(xDecoder1_1)
        xDecoder2_0 = self.deconv_2(xDecoder1_2)
        xDecoder2_1 = self.resunitUp_3(torch.cat([xEncoder1_2, xDecoder2_0],1))
        xDecoder2_2 = self.conv3Up_2(xDecoder2_1)
        xDecoder3_0 = self.deconv_3(xDecoder2_2)
        xDecoder3_1 = self.resunitUp_4(torch.cat([xEncoder0_2, xDecoder3_0],1))
        xDecoder3_2 = self.conv3Up_3(xDecoder3_1)

        # change channel
        output = self.conv1(xDecoder3_2)

        return output



# def unet(pretrained = False, progress = True, *args, **kwargs):
#     model = uNet(*args, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_url, progress=progress)
#         model.load_state_dict(state_dict)
# bhn
#     return model
# #
# if __name__ == '__main__':
#     a = torch.rand((1,1,128,128))
#     net= uNet()
#     dropNum = 0.5
#     net.conv1 = nn.Sequential(nn.Dropout(dropNum), nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
#                 nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                 swish(),)
#     device = ('cuda' if torch.cuda.is_available() else 'cpu')
#     print(device)
#     net = net.to(device = device)
#     a = a.to(device = device)
#     b = net(a)
#     print(b)
