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
from datasetLoaderSN import preBasicDataset
from torchvision import transforms
import torchvision.transforms.functional as F
import PIL.Image as Image
import torch.nn as nn
from shufflenet import ShuffleV2_1_5
from shufflenet import ShuffleV2_2_0
from shufflenet import ShuffleV2_0_5
from shufflenet import ShuffleV2_1_0
from resnet import resnet50
from resnet import resnet101
from resnet import resnet18

xydatapath = './hologramPath/testdata/xyDataInfo.csv'
xyzsavepath = './hologramPath/testdata/'
networkDef = 'ShuffleV2_1_0'#resnet18, resnet50, resnet101, ShuffleV2_1_5, ShuffleV2_2_0, ShuffleV2_0_5,  ShuffleV2_1_0
outputFilePath = './output/expdata_output_noneShuffle_'+networkDef+'/'
networkSavePath = outputFilePath
xyzfile = networkDef + '_predict_xyzDataInfo.csv'
xyzsavepath = xyzsavepath + xyzfile
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model(fine_tune, network):
    if network == 'resnet50':
        net = resnet50(pretrained=False, num_classes=52)
    if network == 'resnet18':
        net = resnet18(pretrained=False, num_classes=52)    
    elif network == 'resnet101':
        net = resnet101(pretrained=False, num_classes=52)    
    elif network == 'ShuffleV2_1_5':
        net = ShuffleV2_1_5(num_classes=52)  
    elif network == 'ShuffleV2_2_0':
        net = ShuffleV2_2_0(num_classes=52)    
    elif network == 'ShuffleV2_1_0':
        net = ShuffleV2_1_0(num_classes=52)  
    elif network == 'ShuffleV2_0_5':
        net = ShuffleV2_0_5(num_classes=52)    
    else:
        print('cannot find the defined network, pls check the name or the lib you imported')
    model = net
    if fine_tune:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False
    return model

fine_tune = True
model = build_model(fine_tune, networkDef).to(device)
model_path=os.path.join(networkSavePath, 'mineUnet_para.pt')
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model'])
net = model.to(device)
net.eval()
    
transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

preHolos = preBasicDataset(xydatapath, transform)

preHoloLoader = torch.utils.data.DataLoader(preHolos, batch_size=1,
                                            shuffle=False)

hologramPath = './hologramPath/testdata/data/'
saveData = []
saveData.append(["fileName","x", "y","z"])
for data in preHoloLoader:
    image = data['image']
    x = data['x']
    x = x.cpu().detach().numpy()[0]
    y = data['y']
    y = y.cpu().detach().numpy()[0]
    id = data['id']
    id = id[0]
    filename = hologramPath + id
    image = image.to(device=device)
    with torch.no_grad():
        outputs = net(image)
    z = torch.max(outputs,1)[1]
    z = z.cpu().detach().numpy()[0]
    saveData.append([filename, x, y, z])

with open(xyzsavepath, "w", newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(saveData)