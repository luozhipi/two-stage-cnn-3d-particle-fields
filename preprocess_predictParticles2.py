import cv2 as cv
import numpy as np
import os
import math

def show_in_one(images, show_size=(300, 600), blank_size=50, window_name="merge"):
    small_h, small_w = images[0].shape[:2]
    column = int(show_size[1] / (small_w + blank_size))
    row = int(show_size[0] / (small_h + blank_size))
    shape = [show_size[0], show_size[1]]
    for i in range(2, len(images[0].shape)):
        shape.append(images[0].shape[i])

    merge_img = np.zeros(tuple(shape), images[0].dtype)

    max_count = len(images)
    count = 0
    for i in range(row):
        if count >= max_count:
            break
        for j in range(column):
            if count < max_count:
                im = images[count]
                t_h_start = i * (small_h + blank_size)
                t_w_start = j * (small_w + blank_size)
                t_h_end = t_h_start + im.shape[0]
                t_w_end = t_w_start + im.shape[1]
                merge_img[t_h_start:t_h_end, t_w_start:t_w_end] = im
                count = count + 1
            else:
                break
    cv.imshow(window_name, merge_img)


def loadImagePath(path):
    ims = []
    ids = []
    for root, dirs, files in os.walk(path):
        for i in files:
            if os.path.splitext(i)[1] == '.tif':
                ims.append(root + '/' + i)
                ids.append(i)
    return ims,ids 

def Laplace(img, kernel):
    des_8U = cv.filter2D(img, -1, kernel=kernel, borderType=cv.BORDER_DEFAULT)
    des_16S = cv.filter2D(img, ddepth=cv.CV_16SC1, kernel=kernel, borderType=cv.BORDER_DEFAULT)
    result = img - des_16S
    #result[result<0] = 0
    #result[result>255] = 255
    return result

def opening_circle(img_bin, kernel_size=10):
	# 形态学
    kernel = np.zeros((kernel_size, kernel_size), np.uint8)
    center_radius = int(kernel_size / 2)
    cv.circle(kernel, (center_radius, center_radius), center_radius, (1, 1, 1), -1, cv.LINE_AA)
    img_open_circle = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)
    return img_open_circle


kernel1 = np.asarray([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])

kernel2 = np.asarray([[1, 1, 1],
                      [1, -8, 1],
                      [1, 1, 1]])
#一个圆形kernel
kernel = np.ones((3, 3), np.uint8)
kernel_re = []
rows, cols = kernel.shape
for i in range(rows):
    result = [0 if math.sqrt((i-3)**2+(j-3)**2) > 3 else 1 for j in range(cols)]
    kernel_re.append(result)
kernel_re = np.array(kernel_re, np.uint8)

particlePath = './hologramPath/testdata/predict/'
preprocess_particlePath = './hologramPath/testdata/post_predict/'
particle, ids = loadImagePath(particlePath)
os.makedirs(preprocess_particlePath, exist_ok = True)
for i in range(len(particle)):
        particle_im = cv.imread(particle[i], 0)
        height = particle_im.shape[0]
        width = particle_im.shape[1]
        colbank = 5#去除verticle边框白色，比如图像边框5个像素
        rowbank = 41#去除horizontal边框白色，比如图像边框5个像素
        for row in range(rowbank):
            for col in range(width):
                particle_im[row, col] = 0
        for row in range(height-rowbank, height):
            for col in range(width):
                particle_im[row, col] = 0
        for col in range(colbank):
            for row in range(height):
                particle_im[row, col] = 0
        for col in range(width-colbank, width):
            for row in range(height):
                particle_im[row, col] = 0                
        #二值化，灰度值大于50就赋予白色，直接取thresh1作为result   
        ret,thresh1 = cv.threshold(particle_im, 100, 255, cv.THRESH_BINARY)
        result = thresh1
        
        #可选一：膨胀操作使得粒子更大
        #dilate = cv.dilate(result, kernel_re, iterations=1)
        #result = dilate

        #可选二：以下片段是为了画出圆形粒子
        #img_open = opening_circle(result, 3)
        #contours, hierarchy = cv.findContours(img_open, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        #result = np.zeros((width, height), dtype=np.uint8)
        #for cnt in contours:
        #    area = cv.contourArea(cnt)#粒子面积
        #    center, radius = cv.minEnclosingCircle(cnt)
        #    circularity = area / (np.pi * radius * radius)#粒子像圆的概率
        #    if circularity > 0.25 and area < 1000:
        #        cv.circle(result, (int(center[0]), int(center[1])), int(radius), (255, 255, 255), -1, cv.LINE_AA)

        #可选三：也可以不做锐化，Laplacian锐化
        #result = Laplace(thresh1, kernel1)
        #imgs = np.hstack([particle_im, thresh1])

        debug_images = []
        debug_images.append(particle_im)
        debug_images.append(result)
        show_in_one(debug_images)
        cv.waitKey(1)#0表所等待输入可以一张一张查看结果，改为1则不需要
        cv.imwrite(preprocess_particlePath + ids[i], result)