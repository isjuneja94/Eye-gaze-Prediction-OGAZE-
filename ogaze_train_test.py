
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
from DataLoader_ik import *


batch=24
trainX = get_frames(vidpath='../gaze_data/train/videos')
traincY,trainfY = get_grids(labels_path='../gaze_data/train/labels',vidpath='../gaze_data/train/videos')

valX = get_frames(vidpath='../gaze_data/val/videos')
valcY,valfY = get_grids(labels_path='../gaze_data/val/labels',vidpath='../gaze_data/val/videos')

testX = get_frames(vidpath='../gaze_data/test/videos')
testcY,testfY = get_grids(labels_path='../gaze_data/test/labels',vidpath='../gaze_data/test/videos')
#print('trainX=',trainX.shape)

trainX = np.rollaxis(trainX,-1,1)
valX = np.rollaxis(valX,-1,1)
testX = np.rollaxis(testX,-1,1)

train_datasetc = VidDataset(trainX,traincY)
val_datasetc = VidDataset(valX,valcY)
test_datasetc = VidDataset(testX,testcY)

#print('normalised? :',np.max(trainX),np.min(trainX),np.mean(trainX[:,0,:,:,:]))
train_datasetf = VidDataset(trainX,trainfY)
val_datasetf = VidDataset(valX,valfY)
test_datasetf = VidDataset(testX,testfY)

train_loaderc = DataLoader(dataset = train_datasetc,batch_size=128,shuffle=True,num_workers=4)
val_loaderc = DataLoader(dataset=val_datasetc,batch_size=128,shuffle=True,num_workers=4)
test_loaderc = DataLoader(dataset=test_datasetc,batch_size=128,shuffle=True,num_workers=4)


train_loaderf = DataLoader(dataset = train_datasetf,batch_size=24,shuffle=True,num_workers=0)
val_loaderf = DataLoader(dataset=val_datasetf,batch_size=24,shuffle=True,num_workers=4)
test_loaderf = DataLoader(dataset=test_datasetf,batch_size=24,shuffle=True,num_workers=4)
epoch = 10
lr = 1e-2
#training the Coarse graze model:
def crossentropy_loss(y_pred,hm)
    hm = hm.squeeze()
    hm = torch.reshape((hm),(hm.shape[0]*hm.shape[1],-1))
    target = torch.argmax(hm,dim=1)
    y_pred = y_pred.squeeze()
    pred = torch.reshape((y_pred),(y_pred.shape[0]*y_pred.shape[1],-1))
    return nn.CrossEntropyLoss(reduction='mean')(pred,target)

#training the fine-graind gaze model:
def MSE_loss(y_pred,hm):
    hm = hm.squeeze()
    hm = torch.reshape((hm),(hm.shape[0]*hm.shape[1],-1))
    #target = torch.argmax(hm,dim=1)
    y_pred = y_pred.squeeze()
    pred = torch.reshape((y_pred),(y_pred.shape[0]*y_pred.shape[1],-1))
    return nn.MSE(pred,hm)

def box_plot_acc(y_pred,y_true):
    y_true = torch.reshape((y_true),(y_true.shape[0]*y_true.shape[1],y_true.shape[2],y_true.shape[3]))
    y_pred = y_pred.squeeze()
    
    y_pred = torch.reshape((y_pred),(y_pred.shape[0]*y_pred.shape[1],y_pred.shape[2],y_pred.shape[3]))
    x0_true = torch.argmax(torch.max(y_true,dim=-2)[0],dim=-1)
    y0_true = torch.argmax(torch.max(y_true,dim=-1)[0],dim=-1)
    x0_pred = torch.argmax(torch.max(y_pred,dim=-2)[0],dim=-1)
    y0_pred = torch.argmax(torch.max(y_pred,dim=-1)[0],dim=-1)
    #print(x0_true.dtype,x0_pred.dtype)
    ss = torch.square(x0_true-x0_pred,)+torch.square(y0_true-y0_pred)
    
    dist = (torch.sqrt(ss.float())).cpu()
    return dist.numpy()

def generate_plots(df):
    history = pd.read_csv(df)
    plt.plot(history['val_loss'],label='val loss')
    plt.plot(history['train_loss'],label='train loss')
    plt.plot(history['test_loss'],label='test loss')
    plt.legend()
    plt.savefig('plots/'+'final_loss_norm_plot'+'.png')
    plt.close()
    plt.plot(history['val_acc'],label='val deviation')
    plt.plot(history['train_acc'],label='train deviation')
    plt.plot(history['test_acc'],label='test deviation')
    plt.legend()
    plt.show()
    plt.savefig('plots/'+'final_acc_norm_plot'+'.png')
    plt.close()
    print('done!')



            
#def combined_loss(CE,MSE):

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"

device = torch.device("cuda")

model_fine = FineAP().to(device)

model_coarse = CoarseAP().to(device)



def train_coarseAP(model,epochs=5,ce_loss = ce_loss):
    history = pd.DataFrame([])
    model = nn.DataParallel(model, device_ids=[8,9])
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    total_train_loss=[]
    total_val_loss=[]
    total_train_acc=[]
    total_val_acc=[]
    total_test_acc=[]
    total_test_loss=[]
    for e in range(epochs):
        loss_epoch = []
        acc_epoch = []
        model.train()
        for batch_idx, (train_X, train_Y) in enumerate(train_loaderc):
            train_X, train_Y = train_X.to(device,dtype = torch.float32), train_Y.to(device,dtype = torch.float32)
            model = Darknet("Yolo9000/cfg/yolo9000.cfg")
            inp = get_test_input(train_X)
            pred = model(inp, torch.cuda.is_available())
            print ("Pred..",pred)
            print('pred shape',pred.shape)
            model = Darknet("Yolo9000/cfg/yolo9000.cfg")
            model.load_weights("Yolo9000weights/yolo9000.weights")


        # zero the parameter gradients
            optimizer.zero_grad()

        # forward + backward + optimize
            output = CourseAP(pred)
            loss = crossentropy_loss(output,train_Y)
            accuracy = cap_acc(output,train_Y)
            loss.backward()
            optimizer.step()
            
            loss_epoch.append(loss.item())
            acc_epoch.append(accuracy.item())
        
        total_train_loss.append(np.mean(loss_epoch))
        total_train_acc.append(np.mean(acc_epoch))
        print(f'Epcoh {e}: ,batch loss:{loss.item()},epoch loss:{np.mean(loss_epoch)},acc:{np.mean(acc_epoch)}',end=' ')
        
        model.eval()
        with torch.no_grad():
            val_loss_epoch=[]
            val_acc_epoch=[]
            for val_X, val_Y in val_loaderc:
                val_X,val_Y = val_X.to(device,dtype = torch.float32),val_Y.to(device,dtype = torch.float32)
                op = CourseAP(val_X)
                val_loss = crossentropy_loss(op,val_Y)
                val_acc = cap_acc(op,val_Y)
                val_loss_epoch.append(val_loss.item())
                val_acc_epoch.append(val_acc.item())
             
            total_val_acc.append(np.mean(val_acc_epoch))    
            total_val_loss.append(np.mean(val_loss_epoch))
            print(f',epoch val_loss:{np.mean(val_loss_epoch)},val_acc={np.mean(val_acc_epoch)}')
            
            
            test_loss_epoch=[]
            test_acc_epoch=[]
            for test_X, test_Y in test_loaderc:
                test_X,test_Y = test_X.to(device,dtype = torch.float32),test_Y.to(device,dtype = torch.float32)
                op = CoarseAP(test_X)
                test_loss = crossentropy_loss(op,test_Y)
                test_acc = cap_acc(op,test_Y)
                test_loss_epoch.append(test_loss.item())
                test_acc_epoch.append(test_acc.item())
             
            total_test_acc.append(np.mean(test_acc_epoch))    
            total_test_loss.append(np.mean(test_loss_epoch))
            print(f',epoch test_loss:{np.mean(test_loss_epoch)},test_acc={np.mean(test_acc_epoch)}')
            
    #torch.save(model.state_dict(),f'models/pytorch_models/modelc_epoch_{epochs}.pth')
    history['train_loss'] = total_train_loss
    history['val_loss'] = total_val_loss
    history['train_acc'] = total_train_acc
    history['val_acc'] = total_val_acc
    history['test_acc'] = total_test_acc
    history['test_loss'] = total_test_loss
    history.to_csv('plots/'+f'history_coarse_{e+1}.csv')
    #generate_plots(f'plots/history_coarse_{e+1}.csv')
    
    



def train_fineAP(model,epochs=1,MSE_loss = MSE_loss):
    history = pd.DataFrame([])
    
    box_df= pd.DataFrame([])    
    model = nn.DataParallel(model, device_ids=[8,9])
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9,weight_decay=1e-2)
    
    total_train_loss=[]
    total_val_loss=[]
    total_train_acc=[]
    total_val_acc=[]
    total_test_acc=[]
    total_test_loss=[]
    print('Training started')
    for e in range(epochs):
        loss_epoch = []
        acc_epoch = []
        box_plot=[]
        model.train()
    
        for batch_idx, (train_X, train_Y) in enumerate(train_loaderf):
            train_X, train_Y = train_X.to(device,dtype = torch.float32), train_Y.to(device,dtype = torch.float32)
            #print((train_X).dtype)
            optimizer.zero_grad()
            output = FineAP(train_X)
            loss = MSE_loss(output, train_Y)
            loss.backward()
            optimizer.step()
            
            loss_epoch.append(loss.item())
            accuracy = fgp_acc(output,train_Y)
            acc_epoch.append(accuracy)       
   
     
        
        total_train_loss.append(np.mean(loss_epoch))
        total_train_acc.append(np.mean(acc_epoch))
       
        print(f'Epcoh {e}: ,batch loss:{loss.item()},epoch loss:{np.mean(loss_epoch)},acc:{np.mean(acc_epoch)}',end=' ')   
    

        model.eval()
        with torch.no_grad():
            val_loss_epoch=[]
            val_acc_epoch=[]
            for val_X, val_Y in val_loaderf:
                val_X,val_Y = val_X.to(device,dtype = torch.float32),val_Y.to(device,dtype = torch.float32)
                op = fineAP(val_X)
                val_loss = MSE_loss(op,val_Y)
                val_acc = fgp_acc(op,val_Y)
                box_plot.extend(list(box_plot_acc(op,val_Y)))
             
                
                val_loss_epoch.append(val_loss.item())
                val_acc_epoch.append(val_acc)
             
            total_val_acc.append(np.mean(val_acc_epoch))    
            total_val_loss.append(np.mean(val_loss_epoch))
            print(f',epoch val_loss:{np.mean(val_loss_epoch)},val_acc={np.mean(val_acc_epoch)}')
            
            
            test_loss_epoch=[]
            test_acc_epoch=[]
            for test_X, test_Y in test_loaderf:
                test_X,test_Y = test_X.to(device,dtype = torch.float32),test_Y.to(device,dtype = torch.float32)
                op = fineAP(test_X)
                test_loss = MSE_loss(op,test_Y)
                test_acc = fgp_acc(op,test_Y)
                test_loss_epoch.append(test_loss.item())
                test_acc_epoch.append(test_acc)

            total_test_acc.append(np.mean(test_acc_epoch))    
            total_test_loss.append(np.mean(test_loss_epoch))
            
        
    box_df[f'{e}'] = box_plot
    torch.save(model.state_dict(),f'modelf_full_norm1_epoch(15/7).pth')
    history['train_loss'] = total_train_loss
    history['val_loss'] = total_val_loss
    history['train_acc'] = total_train_acc
    history['val_acc'] = total_val_acc
    history['test_loss'] = total_test_loss
    history['test_acc'] = total_test_acc
    history.to_csv('plots/'+f'history_fine_norm1(15/7).csv')
    generate_plots(f'plots/history_fine_norm1(15/7).csv')
    box_df.to_csv('box_plots.csv')
    
    
    
    
train_coarseAP(model_coarse,epochs=5,ce_loss=crossentropy_loss)
    
        
    
    