import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision
import torchvision.ops
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils,models
import torch.nn.functional as F

inp = torch.randn(1,3,640,640)

class Yolo(nn.Module):
    def __init__(self,
                 anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
                          (11.2364, 10.0071)],input_channels=3):
        super(Yolo, self).__init__()
        #self.num_classes = num_classes
        self.input_channels=input_channels
        self.anchors = anchors

        self.stage1_conv1 = nn.Sequential(nn.Conv2d(input_channels, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv4 = nn.Sequential(nn.Conv2d(128, 64, 1, 1, 0, bias=False), nn.BatchNorm2d(64),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv5 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv6 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv7 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv8 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv9 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv10 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False), nn.BatchNorm2d(256),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv11 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv12 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False), nn.BatchNorm2d(256),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv13 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                           nn.LeakyReLU(0.1, inplace=True))

        self.stage2_a_maxpl = nn.MaxPool2d(2, 2)
        self.stage2_a_conv1 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False),
                                            nn.BatchNorm2d(1024), nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv2 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False),nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv3 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False),
                                            nn.BatchNorm2d(1024),nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv4 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False),nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv5 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1,bias=False),nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv6 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1,bias=False), nn.BatchNorm2d(1024),nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv7 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1,bias=False), nn.BatchNorm2d(1024),nn.LeakyReLU(0.1, inplace=True))

        self.stage2_b_conv = nn.Sequential(nn.Conv2d(512, 64, 1, 1, 0, bias=False), nn.BatchNorm2d(64),
                                           nn.LeakyReLU(0.1, inplace=True))

        self.stage3_conv1 = nn.Sequential(nn.Conv2d(256 + 1024, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),nn.LeakyReLU(0.1, inplace=True))
        self.stage3_conv2 = nn.Conv2d(1024, 425, 1, 1, 0, bias=False)

    def forward(self, input):
        output = self.stage1_conv1(input)
        output = self.stage1_conv2(output)
        output = self.stage1_conv3(output)
        output = self.stage1_conv4(output)
        output = self.stage1_conv5(output)
        output = self.stage1_conv6(output)
        output = self.stage1_conv7(output)
        output = self.stage1_conv8(output)
        output = self.stage1_conv9(output)
        output = self.stage1_conv10(output)
        output = self.stage1_conv11(output)
        output = self.stage1_conv12(output)
        output = self.stage1_conv13(output)

        residual = output

        output_1 = self.stage2_a_maxpl(output)
        output_1 = self.stage2_a_conv1(output_1)
        output_1 = self.stage2_a_conv2(output_1)
        output_1 = self.stage2_a_conv3(output_1)
        output_1 = self.stage2_a_conv4(output_1)
        output_1 = self.stage2_a_conv5(output_1)
        output_1 = self.stage2_a_conv6(output_1)
        output_1 = self.stage2_a_conv7(output_1)

        output_2 = self.stage2_b_conv(residual)
        batch_size, num_channel, height, width = output_2.data.size()
        #print(batch_size, num_channel, height, width)
        output_2 = output_2.view(batch_size, int(num_channel / 4), height, 2, width, 2).contiguous()
        #print(output_2.shape)
        output_2 = output_2.permute(0, 3, 5, 1, 2, 4).contiguous()
        #print(output_2.shape)
        output_2 = output_2.view(batch_size, -1, int(height / 2), int(width / 2))

        output = torch.cat((output_1, output_2), 1)
        output = self.stage3_conv1(output)
        output = self.stage3_conv2(output)

        return output


#if __name__ == "__main__":
    #net = Yolo(20)
    #print(net.stage1_conv1[0])

#import matplotlib.pyplot as plt
#net(inp)[0][23].shape

class CoarseAP(nn.Module):
    def __init__(self,input_channels):
        super(CoarseAP,self).__init__()

        self.input_channels = input_channels
        self.Yolo = Yolo(self.input_channels)

        self.conv1a = nn.Conv2d(425,512,1)
        self.bn1a = nn.BatchNorm2d(512)
        #self.relu = nn.ReLU()
        
        self.offset_net1 = nn.Conv2d(937,2 * 3 * 3,3)

        self.defconv1b = torchvision.ops.DeformConv2d(937,1,3)
        self.bn1b = nn.BatchNorm2d(1)
        #self.softmax = nn.Softmax(dim=-1)
        
    def forward(self,x):
        

        x1 = self.Yolo(x)

        x2 = self.conv1a(x1)
        x2 = self.bn1a(x2)
        #x2 = self.relu(x2)
        x = torch.cat([x1,x2],axis=1)
        
        offsets1 = self.offset_net1(x)
        x = self.defconv1b(x,offsets1)
        x = self.bn1b(x)
        #x = self.softmax(x)

        return x
    
class FineAP(nn.Module):
    def __init__(self,input_channels):
        super(FineAP,self).__init__()
        self.input_channels = input_channels
        
        self.Yolo = Yolo(self.input_channels)
        
        self.offset_net2 = nn.Conv2d(425,2 * 3 * 3,3)
        self.defconv2a = torchvision.ops.DeformConv2d(425,512,3)
        self.bn2a = nn.BatchNorm2d(512)
        self.pool2a = nn.MaxPool2d(3)
        
        self.offset_net3 = nn.Conv2d(512,2 * 3 * 3,3)
        self.defconv2b = torchvision.ops.DeformConv2d(512,512,3)
        self.bn2b = nn.BatchNorm2d(512)
        self.pool2b = nn.MaxPool2d(3)
        
        self.fl1 = nn.Linear(512*1*1,1024)
        self.fl2 = nn.Linear(1024,2)
        #self.fcl = nn.sequential(nn.linear(16,12),nn.linear(12,10))
        
        #self.MSE = nn.MSELoss(size_average=None, reduce=None, reduction: str = 'mean')
        
    def forward(self,x):
        #start = time.time()
        #x1 = self.exFeat(x)
        #x2 = self.cap(x)
        #x = torch.cat([x1,x2],axis=1)
        
        x = self.Yolo(x)
        offsets2 = self.offset_net2(x)
        x = self.defconv2a(x,offsets2)
        x = self.bn2a(x)
        x = self.pool2a(x)
        
        offsets3 = self.offset_net3(x)
        x = self.defconv2b(x,offsets3)
        x = self.bn2b(x)
        x = self.pool2b(x)
        batch_size =x.shape[0]
        x = torch.reshape(x,(batch_size,-1))
        
        x = self.fl1(x)
        x = self.fl2(x)
        
        return x

def final_model(coarseAP,fineAP):
    model = (coarseAP+fineAP)/2
    return model
    
    
#if __name__=='__main__':    
    #model1 = CoarseAP(3)
    #model2=FineAP(3)
    
        #x = torch.reshape(x,(batch_size,-1))
        #x = nn.linear(x.shape[-1],12)(x)
        #x = nn.linear(12,10)
        #x = self.fcl(x)
        
        #x = self.MSE(x)

        #x = self.conv11b(x)
        #fgp = self.bn11b(x)
        #end = time.time()
        
        


