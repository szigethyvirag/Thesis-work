from cmath import cos
import imp
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cmath



class SesimicDataSet(torch.utils.data.Dataset):
    def __init__(self, length=3000,IsTest=False,do_xy_aug=True, do_fft_aug=False):     

       self.ESeries=np.transpose(np.load('C:\\Users\\Virag\\Desktop\\FÖT\\ESeries.npy'),[0,2,1])
       self.XSeries=np.transpose(np.load('C:\\Users\\Virag\\Desktop\\FÖT\\XSeries.npy'),[0,2,1])
       self.length=int(length)
       self.IsTest=IsTest
       self.do_xy_aug = do_xy_aug
       self.do_fft_aug = do_fft_aug
       if not IsTest:
        self.ESeries = self.ESeries[:500]
        self.XSeries = self.XSeries[:500]
    
    def xy_augmentation(self,Data):
        rand_num =  np.random.random()
        if rand_num<0.25:
            Dat = -1*Data[0,:]
            Data[0,:] = Dat
        elif rand_num<0.5:
            Dat = -1*Data[1,:]
            Data[1,:] = Dat
        elif rand_num<0.75:
            Dat = -1*Data[0,:]
            Data[0,:] = Dat
            Dat = -1*Data[1,:]
            Data[1,:] = Dat


    def fft_augmentation(self,Data):
        Data_fft = np.fft.fft(Data)
        Data_polar_amp = np.abs(Data_fft)
        Data_polar_phase = np.angle(Data_fft, deg = False)

        Drand = np.random.uniform(0.8,1.2,Data.shape)
        Data_polar_amp = Data_polar_amp*Drand

        Shift = np.random.randint(-10,10) / 10
        Data_polar_phase += 2*np.pi*Shift

        Data_fft = Data_polar_amp*np.cos(Data_polar_phase) + Data_polar_amp*np.sin(Data_polar_phase)*1j
        Data = np.fft.ifft(Data_fft).real

    def __len__(self):
        return self.length

    #Noie2noise
    def __getitem__(self, idx): 
        # it is random if the item is an earthquake
        if np.random.random()>0.5: 
            Index = np.random.randint(0,499) 
            if self.IsTest:
                  Index = np.random.randint(2001,self.ESeries.shape[0])   
            Data=self.ESeries[Index,:,:]/np.max(self.ESeries[Index,:,:])
            
            if not self.IsTest and self.do_xy_aug: 
                self.xy_augmentation(Data)
            if not self.IsTest and self.do_fft_aug: 
                self.fft_augmentation(Data)
           
            Label=0
        # or not
        else: 
            Index = np.random.randint(0,499)
            if self.IsTest:
                  Index = np.random.randint(1001,self.XSeries.shape[0])             
            Index = np.random.randint(0,self.XSeries.shape[0])
            Data=self.XSeries[Index,:,:]/np.max(self.XSeries[Index,:,:])
            if not self.IsTest and self.do_xy_aug:
                self.xy_augmentation(Data)
            if not self.IsTest and self.do_fft_aug: 
                self.fft_augmentation(Data)
            Label=1
        
      
        return [ Data,  Label ] 
   


batch_size=10 

train_loader = torch.utils.data.DataLoader(SesimicDataSet(),batch_size=batch_size,drop_last=True)
test_loader = torch.utils.data.DataLoader(SesimicDataSet(IsTest=True),batch_size=batch_size,drop_last=True)


# visualization
#print('label')
#print(next(iter(train_loader))[1][0]) #0 if earthquake, 1 if not
#plt.plot(next(iter(train_loader))[0][0][0],label='x axis') #first element of the first batch x
#plt.plot(next(iter(train_loader))[0][0][1],label='y axis') #first element of the first batch y
#plt.plot(next(iter(train_loader))[0][0][2],label='z axis') #first element of the first batch z
#plt.legend(loc='lower right')
#plt.xlabel('Time [s]')
#plt.ylabel('Normalized magnitude []')
#plt.show()


class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
      
        self.conv1 =torch.nn.Conv1d(3,64,kernel_size=7,stride=3,dilation=1) #64
        self.conv2 =torch.nn.Conv1d(64,128,kernel_size=5,stride=3,dilation=1) #128
        self.conv3 =torch.nn.Conv1d(128,256,kernel_size=3,stride=2,dilation=1) #256
        self.conv4 =torch.nn.Conv1d(256,512,kernel_size=3,stride=2,dilation=1) #512
        self.conv5 =torch.nn.Conv1d(512,512,kernel_size=3,stride=2,dilation=1) #512
        #self.conv6 =torch.nn.Conv1d(512,512,kernel_size=3,stride=2,dilation=1) #512
        self.fc1 = nn.Linear(512*32, 128) #input: fixed -> előző output * 32
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1) #output -> the number of classes (?-1)
    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        
        x = self.conv2(x)
        x = F.leaky_relu(x)
        
        x = self.conv3(x)
        x = F.leaky_relu(x)
        
        x = self.conv4(x)
        x = F.leaky_relu(x)
        
        x = self.conv5(x)
        x = F.leaky_relu(x)

        x= x.view(-1,512*32) 
                
        x=self.fc1(x)
        x = F.leaky_relu(x)
        
        x=self.fc2(x)
        x = F.leaky_relu(x)
        
        x=self.fc3(x)
               
        return x
        
        

UseCuda=True
device = torch.device("cuda" if UseCuda else "cpu")

Net=CNN1D().to(device)

Net.train()
epochs=1000

optimizer = torch.optim.Adam(Net.parameters())
criterion = nn.BCEWithLogitsLoss()
for e in range(epochs):  
      print("Epoch: " + str(e + 1))
      for batch_idx, (Data, Label) in enumerate(train_loader):
              Data = Data.to(device).float()
              Label = Label.to(device).float()
              optimizer.zero_grad()
              response = Net(Data).squeeze()
              loss = criterion(response, Label)
              loss.backward()
              
              optimizer.step()
              if batch_idx%10==0:
                      print("Loss: "+str(loss.item()))
      MeanAcc=0
      for batch_idx, (Data, Label) in enumerate(test_loader):
              Data = Data.to(device).float()
              Label = Label.to(device).float()
              response = F.sigmoid(Net(Data).squeeze())
              MeanAcc+=torch.mean( ((response>0.5)==Label).float()).item()
      torch.save(Net.state_dict(),'seismic_classifier.pth')     
      print("Acc: "+ str(MeanAcc/(batch_idx+1)) )     
