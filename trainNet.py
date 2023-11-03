from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import time
import os
import copy
import torch.utils.data as Data
from torchvision import transforms
from PIL import Image
from PIL import ImageDraw 
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms.functional as F
import unet
import unet_light
import unet_shuffle
import unet_swish
import lossFn
import easyTips as et
from datasetLoader import BasicDataset
from shufflenet import ShuffleV2_1_5

# basic parameters of network
batchSize = 4
learningRate = 0.001
dropNum = 0.5
dropoutFlag = True
validSize = 0.2
epoches = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = 'adam'
newNetwork = True
if newNetwork:
    epochStart = 0
    featureExtract = True
else:
    featureExtract = True #not sure, set a flag here

# lossFn part
imgWidth = 224
imgHeight = 224
lossFn = lossFn.ridgeTV(imgHeight, imgWidth,1, 0.0001)

networkDef = 'twoConv_swish'
holoPath = './hologramPath/dataset/traindata/'
truthPath = './hologramPath/dataset/truth/'
outputFilePath = './output/expdata_output_noneShuffle_'+networkDef+'/'
networkSavePath = outputFilePath

holoImgsList = et.fileList(holoPath)
truthImgsList = et.fileList(truthPath)
os.makedirs(outputFilePath, exist_ok = True)
os.makedirs(networkSavePath, exist_ok = True)

mono = True
showEpochProgress = True
if showEpochProgress:
    epochProgressSaveFolder = './output/expdata_bwbEpochOutput_noneShuffle_lr'+str(learningRate)+'_'+ networkDef+'/'
    os.makedirs(epochProgressSaveFolder, exist_ok=True)

# data load part
class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')

target_image_size = (imgWidth, imgHeight)  # as an example

transform=transforms.Compose([
    SquarePad(),
    transforms.Resize(target_image_size),
    transforms.ToTensor()
])

imagDataset = BasicDataset(holoPath, truthPath, transform)

trainSet, validSet = train_test_split(imagDataset, test_size=validSize, random_state=42)

validDataloader = DataLoader(validSet, batch_size=batchSize, shuffle=True, num_workers=0,
                                             pin_memory=True)
trainDataloader = DataLoader(trainSet, batch_size=batchSize, shuffle=True, num_workers=0,
                                              pin_memory=True)

# network function part
class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

def initialModel(featureExtract, network):
    if network == 'light':
        net = unet_light.uNet()
    elif network == 'shuffle':
        net = unet_shuffle.uNet()
    elif network == 'twoConv':
        net = unet.uNet()
    elif network == 'twoConv_swish':
        net = unet_swish.uNet()
    elif network == 'ShuffleV2_1_5':
        net = ShuffleV2_1_5(num_classes=512)
    else:
        print('cannot find the defined network, pls check the name or the lib you imported')

    model = net
    if featureExtract:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False
    if dropoutFlag:
        model.conv1 = nn.Sequential(nn.Dropout(dropNum), nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                swish(),)

    if not mono:
        model.conv3Down_1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                          nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
                                                         track_running_stats=True),
                                          swish(),)
    return model

def convtImg(img):
    img = img.cpu().detach().numpy()
    underPart = np.max(img) - np.min(img)
    img = (img - np.min(img))/underPart*255
    img= np.squeeze(img)
    img = img.astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

def trainModel(model, trainLoader, validLoader, lossFunc, optimizer,epochFrom, epochTo):
    bestModuleWeight = copy.deepcopy(model.state_dict())
    model = model.to(device=device)
    trainLossHistory = []
    trainTimeWatcher = []
    validTimeWatcher = []
    validLossHistory = []
    epochWatcher = []
    #bestEpochLoss = 0.0
    for epoch in np.arange(epochFrom, epochTo):
        runningLoss = 0.
        model.train()
        for batch in trainLoader:
            inputs = batch['image']
            truth = batch['mask']
            inputs, truth = inputs.to(device=device, dtype=torch.float32), truth.to(device=device, dtype=torch.float32)
            with torch.autograd.set_grad_enabled(True):
                outputs = model(inputs)  # bsize * 2
                loss = lossFunc(outputs, truth)
            runningLoss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()
            runningLoss += loss.item() * inputs.size(0)
        epochLoss = runningLoss / len(trainLoader.dataset)
        print("Epoch: {} Phase: {} loss: {}, time: {}".format(epoch, 'train', epochLoss,
                                                              time.strftime('%d_%H_%M')))  # , file=logFile)

        if showEpochProgress:
            saveName = epochProgressSaveFolder+'train_'+'%03d'%epoch+ '.tif'
            img0 = inputs[0][0]
            img0 = convtImg(img0)
            img1 = outputs[0][0]
            img1 = convtImg(img1)
            img2 = truth[0][0]
            img2 = convtImg(img2)
            dst = Image.new('RGB', (img0.width + img1.width + img2.width + 20, img0.height + 10))
            dst.paste(img0, (0, 0))
            dst.paste(img1, (img0.width + 10 , 0))
            dst.paste(img2, (img0.width + img1.width + 20 , 0))
            draw = ImageDraw.Draw(dst)

            draw.text((0, img0.height),"original",(255,255,255))
            draw.text((img0.width + 10, img0.height),"predict",(255,255,255))
            draw.text((img0.width + img1.width + 20, img0.height),"truth",(255,255,255))
            dst.save(saveName)

        trainLossHistory.append(epochLoss)
        epochWatcher.append(epoch)
        trainTimeWatcher.append(time.strftime('%d_%H_%M'))

        runningLoss = 0.
        model.eval()
        for batch in validLoader:
            inputs = batch['image']
            truth = batch['mask']
            inputs, truth = inputs.to(device=device, dtype=torch.float32), truth.to(device=device, dtype=torch.float32)

            with torch.autograd.set_grad_enabled(False):
                outputs = model(inputs)  # bsize * 2
                loss = lossFn(outputs, truth)

            runningLoss += loss.item() * inputs.size(0)

        epochLoss = runningLoss / len(validLoader.dataset)
        print("Epoch: {} Phase: {} loss: {} time: {}".format(epoch, 'valid', epochLoss,
                                                             time.strftime('%d_%H_%M')))  # , file=logFile)

        if showEpochProgress:
            saveName = epochProgressSaveFolder + 'valid_' + '%03d' % epoch + '.tif'
            img0 = inputs[0][0]
            img0 = convtImg(img0)
            img1 = outputs[0][0]
            img1 = convtImg(img1)
            img2 = truth[0][0]
            img2 = convtImg(img2)
            dst = Image.new('RGB', (img0.width + img1.width + img2.width + 20, img0.height + 10))
            dst.paste(img0, (0, 0))
            dst.paste(img1, (img0.width + 10 , 0))
            dst.paste(img2, (img0.width + img1.width + 20 , 0))
            draw = ImageDraw.Draw(dst)
            draw.text((0, img0.height),"original",(255,255,255))
            draw.text((img0.width + 10, img0.height),"predict",(255,255,255))
            draw.text((img0.width + img1.width + 20, img0.height),"truth",(255,255,255))
            dst.save(saveName)

        validLossHistory.append(epochLoss)
        validTimeWatcher.append(time.strftime('%d_%H_%M'))

        #if epochLoss < bestEpochLoss:
            #bestEpochLoss = epochLoss
        bestModuleWeight = copy.deepcopy(model.state_dict())

    model.load_state_dict(bestModuleWeight)
    titleList = ['epoch', 'loss', 'time']
    trainWrite = [epochWatcher, trainLossHistory, trainTimeWatcher]
    trainLossSavePath = outputFilePath + 'trainLoss_' + '.csv'
    et.csvWriter(titleList, trainWrite, trainLossSavePath)
    validWrite = [epochWatcher, validLossHistory, validTimeWatcher]
    validLossSavePath = outputFilePath + 'validLoss_' + '.csv'
    et.csvWriter(titleList, validWrite, validLossSavePath)
    return model

modelUse = initialModel(featureExtract,networkDef)
print('end to initial the model')
modelUse = modelUse.to(device = device, dtype = torch.float32)
if optimizer == 'adam':
    optimizer = torch.optim.Adam(modelUse.parameters(), lr=learningRate)
elif optimizer == 'RMSprop':
    optimizer = torch.optim.RMSprop(modelUse.parameters(), lr=learningRate, weight_decay=1e-8, momentum=0.9)

# start to train the networ
print('start to train')
modelReturn  = trainModel(modelUse, trainDataloader, validDataloader, lossFn,
                                               optimizer, epochStart, epoches)
paraSave = {'model': modelReturn.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoches}
torch.save(paraSave, networkSavePath +'mineUnet' + '_para.pt')# time.strftime('%m%d%H%M')
torch.cuda.empty_cache()
