
import numpy as np
import random
import matplotlib.pyplot as plt
import os 

from PIL import Image
import PIL.ImageOps  

import torch
from torch.autograd import Variable  
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils

from torch.utils.data import DataLoader, Dataset
from MobileNet.mobilenetV2_model_self import MobilenetV2



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ref: https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb
class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=False):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32)) # 1 = dissimilar, 0 = similar
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
    # def normalize(self, img_arr):
    #     img_arr = img_arr.astype('float32')
    #     for i in range(3):
    #         min_ = img_arr[...,i].min()
    #         max_ = img_arr[...,i].max()
    #         if min_ != max_:
    #             img_arr[...,i] -= min_
    #             img_arr[...,i] *= (255.0/(max_-min_))
    #     return img_arr
    # def get_normalize(self, img):
    #     img_arr = np.array(img)
    #     new_img = Image.fromarray(self.normalize(img_arr).astype('uint8'),'RGB')
    #     return new_img          
    def standardization(self, img):
        img_arr = np.array(img)
        mean = np.mean(img_arr)
        var = np.mean(np.square(img_arr-mean))
        img_arr = (img_arr-mean)/np.sqrt(var)
        new_img = Image.fromarray(img_arr.astype('uint8'), 'RGB')
        return new_img
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        # (1-Y)*0.5*(Dw)^2 + (Y)*0.5*{max(0, m-Dw)}^2
        return loss_contrastive
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3), #１
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*90*90, 500),
            # nn.Dropout(0.5),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            # nn.Dropout(0.5),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    #plt.show()
    plt.savefig('loss.png')
def trans_testing_data(img_path1, img_path2, transform=None, should_invert=False):
    # def normalize(img_arr):
    #     img_arr = img_arr.astype('float32')
    #     for i in range(3):
    #         min_ = img_arr[...,i].min()
    #         max_ = img_arr[...,i].max()
    #         if min_ != max_:
    #             img_arr[...,i] -= min_
    #             img_arr[...,i] *= (255.0/(max_-min_))
    # def get_normalize(img):
    #     img_arr = np.array(img)
    #     new_img = Image.fromarray(normalize(img_arr).astype('uint8'),'RGBA')
    #     return new_img  
    def standardization(img):
        img_arr = np.array(img)
        mean = np.mean(img_arr)
        var = np.mean(np.square(img_arr-mean))
        img_arr = (img_arr-mean)/np.sqrt(var)
        new_img = Image.fromarray(img_arr.astype('uint8'), 'RGB')
        return new_img
    img0 = Image.open(img_path1)
    img1 = Image.open(img_path2)

    if should_invert:
        img0 = PIL.ImageOps.invert(img0)
        img1 = PIL.ImageOps.invert(img1)


    if transform is not None:
        img0 = transform(img0)
        img1 = transform(img1)
    return img0, img1

def testing(img_path1, img_path2, model_path):
    test_dataset = trans_testing_data(img_path1=img_path1, 
                                    img_path2=img_path2, 
                                    transform=transforms.Compose([transforms.ToTensor()#,  
                                        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ]),
                                    should_invert=False)
    test_dataloader = DataLoader(test_dataset,
                        shuffle=True, 
                        num_workers=8,
                        batch_size=1)
    #net = SiameseNetwork().to(device)
    net = MobilenetV2().to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    criterion = ContrastiveLoss()
    img0, img1 = test_dataloader
    img0, img1 = img0.to(device), img1.to(device)
    
    #output1, output2 = net(img0, img1)
    output1 = net(img0)
    output2 = net(img1)
    euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
    
    return euclidean_distance.item()
def training(training_dir, train_batch_size, train_epochs):
    folder_dataset = dset.ImageFolder(root = training_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset, 
                                            transform=transforms.Compose([
                                                                    transforms.ToTensor()#,
                                                                    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                                    ])
                                       ,should_invert=False)
    # siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset, 
    #                                         transform=transforms.Compose([transforms.Resize((100,100)),
    #                                                                   transforms.ToTensor()
    #                                                                   ])
    #                                    ,should_invert=False)
    train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=train_batch_size)
    # net = SiameseNetwork().to(device)
    net = MobilenetV2().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr= 0.0005)
    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(0, train_epochs):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            ##### clear all params grad 清除所有參數梯度緩存
            optimizer.zero_grad()
            # output1, output2 = net(img0, img1)
            output1 = net(img0)
            output2 = net(img1)
            # print('output1, output2: ' , output1, output2)
            loss_c = criterion(output1, output2, label)
            loss_c.backward()
            optimizer.step()
            if i %10 == 0:
                print('Epoch number {}, loss {}'.format(epoch, loss_c.item() ))
                model_path = os.path.join(model_root, str(epoch)+'_' +str(loss_c.item()) + '.pt')
                torch.save(net.state_dict(),model_path)
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_c.item())
    show_plot(counter,loss_history)
def predict_test(img_ok_path1, img_path, model_path):
    
    predict = testing(img_ok_path1, img_path, model_path)
    return 0 if predict <=0.5 else 1 ## ok = 0, ng = 1
    
