from sklearn.model_selection import train_test_split
import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
import easyTips as et
from tqdm import tqdm
import matplotlib
#matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
from shufflenet import ShuffleV2_1_5
from shufflenet import ShuffleV2_2_0
from shufflenet import ShuffleV2_0_5
from shufflenet import ShuffleV2_1_0
from resnet import resnet50
from resnet import resnet101
from resnet import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(networkSavePath+'/accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(networkSavePath+'/loss.png')

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

from torch.autograd import Variable
def train(model, trainloader, optimizer, criterion):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    with torch.autograd.set_grad_enabled(True):
        for batch_idx, (image, labels) in enumerate(trainloader):
            image, labels = Variable(image).to(device), Variable(labels).to(device)
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, labels)
            train_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == labels).sum().item()
            loss.backward()
            optimizer.step()

    epoch_loss = train_running_loss / len(trainloader)
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

def validate(model, testloader, criterion):
    model.eval()
    valid_running_loss = 0.0
    valid_running_correct = 0
    with torch.autograd.set_grad_enabled(False):
        for batch_idx, (image, labels) in enumerate(testloader):
            image, labels = Variable(image).to(device), Variable(labels).to(device)
            outputs = model(image)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / len(testloader)
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

transform=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),    
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

networkDef = 'ShuffleV2_1_0'#resnet18, resnet50, resnet101, ShuffleV2_1_5, ShuffleV2_2_0, ShuffleV2_0_5,  ShuffleV2_1_0
outputFilePath = './output/expdata_output_noneShuffle_'+networkDef+'/'
networkSavePath = outputFilePath
root_dir = './hologramPath/sndataset/traindata/'
os.makedirs(networkSavePath, exist_ok = True)

batch_size = 8
valid_split = 0.2
lr = 0.001
epochs = 20
fine_tune = True

dataset = datasets.ImageFolder(root_dir, transform=transform)
dataset_size = len(dataset)
print(f"Total number of images: {dataset_size}")
valid_size = int(valid_split*dataset_size)
train_size = len(dataset) - valid_size
# training and validation sets
train_data, valid_data = torch.utils.data.random_split(
    dataset, [train_size, valid_size]
)
print(f"Total training images: {len(train_data)}")
print(f"Total valid_images: {len(valid_data)}")
# training and validation data loaders
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=0
)
valid_loader = DataLoader(
    valid_data, batch_size=batch_size, shuffle=True, num_workers=0
)
model = build_model(fine_tune, networkDef).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(networkDef)
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.\n")
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# loss function
criterion = nn.CrossEntropyLoss()

# start to train the networ
#print('start to train')
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    newlr = lr * (0.1 ** (epoch // 10))
    print("lr:", newlr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = newlr
# lists to keep track of losses and accuracies
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
# start the training
for epoch in range(epochs):
    adjust_learning_rate(optimizer, epoch)
    train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                              optimizer, criterion)
    valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                 criterion)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Epoch {epoch+1} of {epochs} Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Epoch {epoch+1} of {epochs} Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    print('-'*50)
#save_plots(train_acc, valid_acc, train_loss,valid_loss)
# save network and parameters
titleList = ['loss']
trainWrite = [train_loss]
trainLossSavePath = outputFilePath + 'trainLoss_' + '.csv'
et.csvWriter(titleList, trainWrite, trainLossSavePath)
validWrite = [valid_loss]
validLossSavePath = outputFilePath + 'validLoss_' + '.csv'
et.csvWriter(titleList, validWrite, validLossSavePath)

titleList = ['acc']
trainWrite = [train_acc]
trainAccSavePath = outputFilePath + 'trainAcc_' + '.csv'
et.csvWriter(titleList, trainWrite, trainAccSavePath)
validWrite = [valid_acc]
validAccSavePath = outputFilePath + 'validAcc_' + '.csv'
et.csvWriter(titleList, validWrite, validAccSavePath)

paraSave = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epochs}
torch.save(paraSave, networkSavePath +'mineUnet' + '_para.pt')#time.strftime('%m%d%H%M') + 
torch.cuda.empty_cache()


#train loss 下降↓，val loss 下降 ↓：训练正常，网络仍在学习，最好的情况。
#train loss 下降 ↓，val loss：上升/不变：有点过拟合overfitting，可以停掉训练，用过拟合方法如数据增强、正则、dropout、max pooling等。
#train loss 稳定，val loss 下降：数据有问题，检查数据标注有没有错，分布是否一直，是否shuffle。
#train loss 稳定，val loss 稳定：学习过程遇到瓶颈，可以尝试调小学习率或batch数量
#train loss 上升 ↑，val loss 上升 ↑：网络结构设计不当，参数不合理，数据集需清洗等，最差情况。