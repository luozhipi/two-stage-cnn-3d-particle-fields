参考:shufflenet分类网络 https://zhuanlan.zhihu.com/p/440217402

运行顺序:
step 1:mineSimuGen.py 
生成训练数据集，包括全息粒子图像（分辨率为512*512），和粒子真值图 （分辨率为512*512），以及csv文件记录图像中粒子的x, y, z
文件目录为
isaveFilePath = './hologramPath/dataset/traindata/' ：图像
psaveFilePath = './hologramPath/dataset/truth/' :粒子真值 
tsaveFilePath = './hologramPath/dataset/' ：存放噪声模板文件template01和template02的目录
zsaveFilePath = './hologramPath/dataset/' ：存放csv的目录
saveCsvFileName = 'trainDataInfo.csv'

step 2:trainNet.py 
训练xy预测网络unet-swish， 输入网络的图像分辨率为(224, 224)

step 3：predictNet.py
预测粒子的xy值
文件目录为:
hologramPath = './hologramPath/testdata/data/' ：测试图像，实拍图像为1920*1200分辨率, 因此resize为方形图像时候就会使用squarpad，或opencv的copyMakeBorder填充
outputPath = './output/results/' ：存放临时结果的可视化
outputParticlePath = './hologramPath/testdata/predict/'：预测的粒子xy值图像分辨率为(224, 224)

setp 4：preprocess_predictParticles2.py
后处理，最终结果保存在'./hologramPath/testdata/post_predict/'

step 5: mineSimuGen2.py 
particlePerImg = 1,重新生成每张图像只有一个粒子的图像来生成网络2所需的数据集，这样粒子就不会重叠

step 6: holoParticle2.py
process_for_training()函数是从step 1中生成对应的粒子Z轴预测网络所需的训练数据集
文件目录为：    
savedir = './hologramPath/sndataset/traindata/': 单一粒子图像分辨率(128*128)
zsavepath = './hologramPath/sndataset/trainDataInfo.csv'：记录图像对应的z值

step 7： trainNetSN.py
训练预测粒子Z轴的shufflenet网络

step 8: holoParticle.py
process_from_swishnet_output()函数是根据step 3中unet-swish网络预测的粒子xy中提取出单一粒子图
文件目录为:
savedir = './hologramPath/testdata/dataSN/'：从测试图像目录testdata/data/提取单一粒子图像分辨率(128*128)
xysavepath = './hologramPath/testdata/xyDataInfo.csv' 记录图像对应的xy值

step 9：predictNetSN.py
预测粒子Z值
文件目录为：
xyzsavepath = './hologramPath/testdata/predict_xyzDataInfo.csv' 保存最终的结果，记录testdata/data目录下测试图像对应的x, y, z值