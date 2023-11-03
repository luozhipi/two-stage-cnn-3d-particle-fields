import numpy as np
from math import pi
import os
import cv2 as cv
import random
import csv
import numba as nb
from skimage.util import random_noise
import numpy as np

waveLength = 450*10e-6
pixelSize = 3.69*10e-3


#isaveFilePath = './hologramPath/testdata/data/'
#tsaveFilePath = './hologramPath/testdata/truth/'
#zsaveFilePath = './hologramPath/'
#saveCsvFileName = 'testDataInfo.csv'
#bgIntensity = 2
#imgNum = 10

isaveFilePath = './hologramPath/dataset/traindata/'
psaveFilePath = './hologramPath/dataset/truth/'
tsaveFilePath = './hologramPath/dataset/'
zsaveFilePath = './hologramPath/dataset/'
saveCsvFileName = 'trainDataInfo.csv'
bgIntensity = 2
imgNum = 1000

if os.path.isdir(isaveFilePath) == False:
    os.mkdir(isaveFilePath)
if os.path.isdir(tsaveFilePath) == False:
    os.mkdir(tsaveFilePath)
if os.path.isdir(psaveFilePath) == False:
    os.mkdir(psaveFilePath)    
fixXY = False
xMax = 512
yMax = 512


fixZ = False
zMin = 3#7#37
zStep = 0.05#0.1#0.4
zLayers = 52#4#512
zHolo = None

fixSize = False
sHolo = 15
sMin = 7.5#None
sMax = 9.5#None


imgName = []
xCoordinate = []
yCoordinate = []
zCoordinate = []
zReal = []
sizeDistribution = []

@nb.jit(nopython = True, parallel = True)
def logicalMat(mat, thre):
    for x in np.arange(mat.shape[0]):
        for y in np.arange(mat.shape[1]):
            if mat[x,y] <= thre:
                mat[x,y] = 0
            else:
                mat[x,y] = 1
    return mat

def saveImg(imgName, imgMat, template01, template02):
    imgMat = 255*(imgMat - np.min(imgMat))/(np.max(imgMat) - np.min(imgMat))
    imgMat = np.uint8(imgMat)
    #cv.imshow('original', imgMat)

    # Add salt-and-pepper noise to the image.
    #amount = np.random.uniform(0.3, 0.8)
    #noise_img = random_noise(imgMat, mode='s&p',amount=amount)
    #noise_img = np.array(255*(1.0-noise_img), dtype = 'uint8')
    #gauss = np.random.normal(0,3,imgMat.size)
    #gauss = gauss.reshape(imgMat.shape[0],imgMat.shape[1]).astype('uint8')
    #img_gauss = cv.add(imgMat,gauss)
    #imgMat = img_gauss
    
    alpha = np.random.uniform(0.3, 0.5)
    beta = (1.0 - alpha)
    selection = np.random.randint(0, 2)
    if selection==0:
        imgMat = cv.addWeighted(imgMat, alpha, template01, beta, 0.0)
    elif selection==1:
        imgMat = cv.addWeighted(imgMat, alpha, template02, beta, 0.0)
    else:
        imgMat = imgMat
    #cv.imshow('real',imgMat)
    #cv.waitKey(1)
    cv.imwrite(imgName, imgMat)

def save_particle_img(img_particle_Name, img_particle):
    cv.imwrite(img_particle_Name, img_particle)

k = 2*pi/waveLength
template01 = cv.imread(tsaveFilePath + 'template01.tif', cv.IMREAD_GRAYSCALE)
template01 = cv.resize(template01, (xMax, yMax))
template02 = cv.imread(tsaveFilePath + 'template02.tif', cv.IMREAD_GRAYSCALE)
template02 = cv.resize(template02, (xMax, yMax))
for imgIndex in np.arange(imgNum): 
    holoPic = np.zeros((2 * yMax, 2 * xMax), dtype=np.float32)
    img_particle = np.zeros((yMax, xMax), dtype=np.uint8)
    particlePerImg = random.randint(30, 55)
    #particlePerImg = 1 #一个粒子，为训练预测Z值网络生成训练数据
    for particleIndex in np.arange(particlePerImg):
        if fixZ:
            pass
        else:
            zHoloIndex = np.random.randint(0, zLayers)
            zHolo = zHoloIndex*zStep+zMin

        if fixXY:
            pass
        else:
            xHolo = np.random.randint(0, xMax)
            yHolo = np.random.randint(0, yMax)

        if fixSize:
            pass
        else:
            sHolo = np.random.randint(sMin, sMax)
        fx = np.arange(-1/(pixelSize), 1/(pixelSize), 1 / (xMax * pixelSize))
        fy = np.arange(-1/(pixelSize), 1/(pixelSize), 1 / (yMax * pixelSize))
        [FX, FY] = np.meshgrid(fx, fy)
        cv.circle(img_particle, (xHolo, yHolo), 4, (255, 255, 255), -1)
        #cv.putText(img_particle, str('{:.2f}'.format(zHolo)), (xHolo, yHolo), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv.LINE_AA)
        temp = sHolo * sHolo / np.square(xMax * pixelSize)
        obj1 = np.square(FX-FY+yHolo/(yMax*pixelSize)-xHolo/(xMax*pixelSize))# * (FX-FY+yHolo/(yMax*pixelSize)-xHolo/(xMax*pixelSize))
        obj2 = np.square(FX+FY-yHolo/(yMax*pixelSize)-xHolo/(xMax*pixelSize)+2/pixelSize) #* (FX+FY-yHolo/(yMax*pixelSize)-xHolo/(xMax*pixelSize))
        obj = obj1 / temp + obj2 / temp
        obj = logicalMat(obj, 1)  # define the particle
        f1 = np.square(FX)+np.square(FY)
        imgFft = np.fft.fft2(np.fft.fftshift(obj))
        H = np.fft.fftshift(np.exp(2*1j * zHolo * pi * waveLength * f1))
        outputImgTemp = np.fft.ifftshift(np.fft.ifft2(imgFft * H))
        holoPic = holoPic + outputImgTemp

        xCoordinate.append(xHolo)
        yCoordinate.append(yHolo)
        zCoordinate.append(zHoloIndex)
        zReal.append(zHolo)
        sizeDistribution.append(sHolo)
        holoName = 'holo_%04d.tif' % imgIndex
        holoName = isaveFilePath + holoName

        imgName.append(holoName)

    bg = bgIntensity * np.ones(holoPic.shape)
    holo2 = holoPic + bg
    holomin2 = np.min(holo2)
    holomax2 = np.max(holo2)

    holo2 = (holo2 - holomin2) / (1.15 * (holomax2 - holomin2))
    holo2 = (holo2 + 0.1) / (1 + 0.12) * 255
    holo2 = holo2[0:yMax, 0:xMax]
    holo2 = np.abs(holo2)
    saveImg(holoName, holo2, template01, template02)

    img_particle_Name = psaveFilePath + 'holo_%04d.tif' % imgIndex
    save_particle_img(img_particle_Name, img_particle)
    if imgIndex%100==0:
        print('gened '+str(imgIndex))

csvSavePath = zsaveFilePath+saveCsvFileName
with open(csvSavePath, 'w', newline='') as testFile:
    testWrite = csv.writer(testFile)
    testWrite.writerow(('fileName', 'size', 'x', 'y', 'z', 'zReal'))
    for imgCount in np.arange(len(imgName)):
        testWrite.writerow((imgName[imgCount], sizeDistribution[imgCount], xCoordinate[imgCount],yCoordinate[imgCount], zCoordinate[imgCount], zReal[imgCount]))
