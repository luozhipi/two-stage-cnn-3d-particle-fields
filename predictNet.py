import torch
# import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import easyTips as et
from myUnet import uNet
import unet
import unet_swish
import unet_shuffle
import unet_light
import numpy as np
from datasetLoader import preBasicDataset
import PIL.Image as Image
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import ImageDraw 
# Basic directory parameters
hologramPath = './hologramPath/testdata/data/'
outputPath = './output/results/'
outputParticlePath = './hologramPath/testdata/predict/'
# outputName = outputPath+

mono = True
# Basic net parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
networkDef = 'twoConv_swish'
outputFilePath = './output/expdata_output_noneShuffle_'+networkDef+'/'
networkSavePath = outputFilePath

class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x
    
def convtImg(img):
    img = img.cpu().detach().numpy()
    underPart = np.max(img) - np.min(img)
    img = (img - np.min(img))/underPart*255
    img= np.squeeze(img)
    img = img.astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

def initialModel(network):
    if network == 'light':
        net = unet_light.uNet()
    elif network == 'shuffle':
        net = unet_shuffle.uNet()
    elif network == 'twoConv':
        net = unet.uNet()
    elif network == 'twoConv_swish':
        net = unet_swish.uNet()
    else:
        print('cannot find the defined network, pls check the name or the lib you imported')

    model = net

    model.conv1 = nn.Sequential(nn.Dropout(0.5), nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            swish(),)

    if not mono:
        model.conv3Down_1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                          nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
                                                         track_running_stats=True),
                                          swish(),)
    return model

# data load part
class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')


target_image_size = (224, 224)  # as an example
transform=transforms.Compose([
    SquarePad(),
    transforms.Resize(target_image_size),
    transforms.RandomEqualize(p=1.0),#可以对实拍测试图像做一些预处理
    transforms.RandomAutocontrast(p=1.0),#可以对实拍测试图像做一些预处理
    transforms.ToTensor()
])
preHolos = preBasicDataset(hologramPath,transform)
fileList = et.fileList(hologramPath)
preHoloLoader = torch.utils.data.DataLoader(preHolos, batch_size=1,
                                            shuffle=False, num_workers=0, pin_memory=True)


model = initialModel(network = 'twoConv_swish')
model_path = os.path.join(networkSavePath, 'mineUnet_para.pt')
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()

os.makedirs(outputPath, exist_ok = True)
os.makedirs(outputParticlePath, exist_ok = True)
for batch in preHoloLoader:
    inputs = batch['image']
    filename = batch['id']
    inputs = inputs.to(device = device)
    with torch.no_grad():
        outputs = model(inputs)
    img0 = convtImg(inputs[0][0])    
    img1 = convtImg(outputs[0][0])
    img1.save(outputParticlePath + filename[0])

    img0=img0.resize((640, 640))
    img1=img1.resize((640, 640))

    dst = Image.new('RGB', (img0.width + img1.width + 10, img0.height + 10))
    dst.paste(img0, (0, 0))
    dst.paste(img1, (img0.width + 10 , 0))
    draw = ImageDraw.Draw(dst)
    draw.text((0, img0.height),"original",(255,255,255))
    draw.text((img0.width + 10, img0.height),"predict",(255,255,255))
    preSaveName = outputPath + 'predict_'+filename[0]
    dst.save(preSaveName)
