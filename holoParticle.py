import csv
import cv2 as cv
import numpy as np
import pandas as pd
import os

def loadImagePath(path):
    ims = []
    for root, dirs, files in os.walk(path):
        for i in files:
            if os.path.splitext(i)[1] == '.tif':
                ims.append(root + '/' + i)
    return ims  

def findSpot(im):
    ret, imb = cv.threshold(im, 127, 255, cv.THRESH_BINARY)
    ret, labels, stats, centroid = cv.connectedComponentsWithStats(imb)
    particles = []
    for i, stat in enumerate(stats):
        cv.rectangle(im, (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3]), (25, 25, 255), 3)
        cv.putText(im, str(i + 1), (stat[0], stat[1] + 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 25, 25), 2)
        cv.circle(im, (stat[0]+stat[2]//2, stat[1] + stat[3]//2), 10, (0,0,255))
        particles.append([stat[0]+stat[2]//2, stat[1] + stat[3]//2])
    return particles[1:]


def splitHolo(im, pts, halfWin, saveDir, imgIdx):
    # win = 64
    fullWin = halfWin * 2
    w, h = im.shape
    cnt = 0
    imName = []
    idName = []
    for p in pts:
        p1, p2, p3, p4 = p[0]-halfWin, p[0]+halfWin, p[1]-halfWin, p[1]+halfWin
        if p1 < 0:
            p1, p2 = 0, fullWin
        if p2 >= w:
            p1, p2 = w-1-fullWin, w-1
        if p3 < 0:
            p3, p4 = 0, fullWin
        if p4 >= h:
            p3, p4 = h-1-fullWin, h-1
        iim = im[p3:p4, p1:p2]
        #cv.imshow('swishpredict', iim)
        #cv.waitKey(1)
        filename = saveDir + imgIdx +'_' + str(cnt) +'.tif'
        cv.imwrite(filename, iim)
        imName.append(filename)
        idName.append(imgIdx)
        cnt+=1
    return imName, idName

def splitHolo_for_training(im, pts, halfWin, saveDir, imgIdx):
    # win = 64
    fullWin = halfWin * 2
    w, h = im.shape
    cnt = 0
    imName = []
    for p in pts:
        p1, p2, p3, p4 = p[0]-halfWin, p[0]+halfWin, p[1]-halfWin, p[1]+halfWin
        if p1 < 0:
            p1, p2 = 0, fullWin
        if p2 >= w:
            p1, p2 = w-1-fullWin, w-1
        if p3 < 0:
            p3, p4 = 0, fullWin
        if p4 >= h:
            p3, p4 = h-1-fullWin, h-1
        iim = im[p3:p4, p1:p2]
        #cv.imshow('traning', iim)
        #cv.waitKey(1)
        filename = saveDir + 'holoParticle_' + imgIdx +'_' + str(p[0]) + '_' + str(p[1]) +'.tif'
        cv.imwrite(filename, iim)
        imName.append(filename)
        cnt+=1
    return imName

def  process_from_swishnet_output():
    holoPath = './hologramPath/testdata/data/'
    particlePath = './hologramPath/testdata/post_predict/'
    savedir = './hologramPath/testdata/dataSN/'
    xysavepath = './hologramPath/testdata/xyDataInfo.csv'
    os.makedirs(savedir, exist_ok = True)
    particle = loadImagePath(particlePath)
    holo = loadImagePath(holoPath)
    halfWin = 16#64
    saveData = []
    saveData.append(["fileName","imgID", "x","y"])
    for i in range(len(particle)):
        particle_im = cv.imread(particle[i], 0)
        particle_im = cv.resize(particle_im, (512,512))
        pts = findSpot(particle_im)
        imgIdx = particle[i].split('/')[-1][0:-4]
        holo_im = cv.imread(holo[i],0)
        holo_im = cv.copyMakeBorder(holo_im,360,360,0,0,cv.BORDER_CONSTANT,value = 0) #针对实拍原图分辨率为1920*1200
        holo_im = cv.resize(holo_im, (512,512))
        img_names,idNames = splitHolo(holo_im, pts, halfWin, savedir, imgIdx)
        #cv.imshow('hu', particle_im)
        #cv.imshow('kk', holo_im)
        #cv.waitKey(1)
        for j in range(len(pts)):
            saveData.append([img_names[j], idNames[j], pts[j][0],pts[j][1]])
    with open(xysavepath, "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(saveData)

def process_for_training():
    datainfopath = './hologramPath/dataset/trainDataInfo.csv'
    zvalues = pd.read_csv(datainfopath)['z']
    xvalues = pd.read_csv(datainfopath)['x']
    yvalues = pd.read_csv(datainfopath)['y']
    holofilenames = pd.read_csv(datainfopath)['fileName']
    savedir = './hologramPath/sndataset/traindata/'
    zsavepath = './hologramPath/sndataset/trainDataInfo.csv'
    os.makedirs(savedir, exist_ok = True)
    halfWin = 64
    saveData = []
    saveData.append(["fileName","z"])
    for i in range(len(holofilenames)):
        holo_im = cv.imread(holofilenames[i], 0)
        imgIdx = holofilenames[i].split('/')[-1][0:-4]
        x = np.array(xvalues)[i]
        y = np.array(yvalues)[i]
        z = np.array(zvalues)[i]
        pts=[[x,y]]
        img_names = splitHolo_for_training(holo_im, pts, halfWin, savedir, imgIdx)
        for j in range(len(pts)):
            saveData.append([img_names[j],z])
    with open(zsavepath, "w", newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerows(saveData)   

def main():
    process_from_swishnet_output()
    #process_for_training()

if __name__ == '__main__':
    main()
