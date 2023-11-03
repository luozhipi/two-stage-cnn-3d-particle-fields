import easyTips as et
import cv2 as cv
import numpy as np
import os

filePath = r'/media/hololab/72a309a5-6fab-4ddb-8901-596a2ecaf3f7/buyu/uNet/dg4particleXY/'
outputPath = r'/media/hololab/72a309a5-6fab-4ddb-8901-596a2ecaf3f7/buyu/uNet/bwb/'
outputPath2 = r'/media/hololab/72a309a5-6fab-4ddb-8901-596a2ecaf3f7/buyu/uNet/bww/'

os.makedirs(outputPath, exist_ok=True)
os.makedirs(outputPath2, exist_ok=True)

fileL = et.fileList(filePath)

for x in np.arange(len(fileL)):
    img = cv.imread(fileL[x],0)
    ret, bwb = cv.threshold(img, 3, 255, cv.THRESH_BINARY)
    ret, bww = cv.threshold(img, 3, 255, cv.THRESH_BINARY_INV)
    saveName1 = outputPath + os.path.split(fileL[x])[1]
    saveNAme2 = outputPath2 + os.path.split(fileL[x])[1]
    cv.imwrite(saveName1, np.uint8(bwb) )
    cv.imwrite(saveNAme2, np.uint8(bww) )



