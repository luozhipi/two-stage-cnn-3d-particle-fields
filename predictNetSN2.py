import torch
# import torch.nn.functional as F
import os
import csv
import matplotlib.pyplot as plt
import easyTips as et
from myUnet import uNet
import unet
import unet_swish
import unet_shuffle
import unet_light
import numpy as np
from datasetLoaderSN import BasicDataset2
from torchvision import transforms
import torchvision.transforms.functional as F
import PIL.Image as Image
import torch.nn as nn
from shufflenet import ShuffleV2_1_5

zdatapath = './hologramPath/sndataset/trainDataInfo.csv'
zsavepath = './hologramPath/sndataset/predict_zDataInfo.csv'
networkDef = 'ShuffleV2_1_5'
outputFilePath = './output/expdata_output_noneShuffle_'+networkDef+'/'
networkSavePath = outputFilePath
mono = True
# Basic net parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

def initialModel(network):
    if network == 'light':
        net = unet_light.uNet()
    elif network == 'shuffle':
        net = unet_shuffle.uNet()
    elif network == 'twoConv':
        net = unet.uNet()
    elif network == 'twoConv_swish':
        net = unet_swish.uNet()
    elif network == 'ShuffleV2_1_5':
        net = ShuffleV2_1_5(num_classes=52)
    else:
        print('cannot find the defined network, pls check the name or the lib you imported')

    model = net
    if not mono:
        model.conv3Down_1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                          nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
                                                         track_running_stats=True),
                                          swish(),)
    return model

model = initialModel(network = 'ShuffleV2_1_5')
model_path=os.path.join(networkSavePath, 'mineUnet_para.pt')
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model'])
net = model.to(device)
net.eval()

target_image_size = (32, 32) 
# data load part
class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')
    
transform=transforms.Compose([
    SquarePad(),
    transforms.Resize(target_image_size),
    transforms.ToTensor()
])

preHolos = BasicDataset2(zdatapath, transform)

preHoloLoader = torch.utils.data.DataLoader(preHolos, batch_size=1,
                                            shuffle=False, num_workers=0, pin_memory=True)

saveData = []
saveData.append(["fileName", "z_truth", "z"])
for b in preHoloLoader:
    inputs = b['image']
    id = b['id']
    filename = id[0]
    z_truth = b['z_truth']
    z_truth = z_truth.cpu().detach().numpy()[0]
    inputs = inputs.to(device=device)
    with torch.no_grad():
        out = net(inputs)
    z = torch.max(out,1)[1]
    z = z.cpu().detach().numpy()[0]
    saveData.append([filename, z_truth, z])

with open(zsavepath, "w", newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(saveData)