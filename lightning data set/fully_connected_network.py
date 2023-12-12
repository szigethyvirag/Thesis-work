import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
import geopy
import geopy.distance

class LightningDataSet(torch.utils.data.Dataset):
    def __init__(self,sampleLength = 5, IsTest=False):

        self.Series = np.transpose(np.load('series_array.npy'),[0,2,1])
        self.length=int(self.Series.shape[0])
        self.IsTest=IsTest
        self.splitPoint = int(np.floor(self.length * 0.8))
        self.sampleLength = sampleLength 
        if not IsTest:
            self.Series = self.Series[:self.splitPoint]

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        selectedData = np.array([[0]])
        while selectedData.shape[1] < self.sampleLength + 1: 

            # choose an index -> which serie we want
            Index = np.random.randint(0,self.splitPoint-1) 
            if self.IsTest:
                Index = np.random.randint(self.splitPoint,self.length-1)  
            fullData=self.Series[Index,:,:] 
            selectedData = fullData[:, ~np.isinf(fullData).any(axis=0)]
   
        if selectedData.shape[1] == self.sampleLength+1: 
            randomStart = 0
        else:
            randomStart = np.random.randint(0, selectedData.shape[1]-(self.sampleLength+1)) 
        Data = selectedData[:,randomStart:randomStart+self.sampleLength] 
        Label = selectedData[:,randomStart+self.sampleLength] 
        return [ Data,  Label ] 
        
        
batch_size = 100 
coords = 2
sampleLength = 10 

train_loader = torch.utils.data.DataLoader(LightningDataSet(sampleLength),batch_size=batch_size,drop_last=True)
test_loader = torch.utils.data.DataLoader(LightningDataSet(sampleLength, IsTest=True),batch_size=batch_size,drop_last=True)


class FullyConnectedRegression(nn.Module):
    def __init__(self):
        super(FullyConnectedRegression, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(coords*sampleLength, 64)  
        self.fc2 = nn.Linear(64, coords*1) 

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        
        return x

def my_criterion(response, Label):
    response = response.view(-1,2)
    lat1, lon1 = response[:, 0], response[:, 1]
    lat2, lon2 = Label[:, 0], Label[:, 1]

    # Convert latitude and longitude to tuples
    coords1 = list(zip(lat1.tolist(), lon1.tolist()))
    coords2 = list(zip(lat2.tolist(), lon2.tolist()))

    # Calculate geodesic distances
    distances = [geopy.distance.geodesic(coord1, coord2).km for coord1, coord2 in zip(coords1, coords2)]
    loss = torch.mean((torch.tensor(distances)))
    return loss
        

UseCuda=True
device = torch.device("cuda" if UseCuda else "cpu")

Net=FullyConnectedRegression().to(device)

Net.train()
epochs=1000

optimizer = torch.optim.Adam(Net.parameters())
criterion = nn.MSELoss()
for e in range(epochs):  
      print("Epoch: " + str(e + 1))
      for batch_idx, (Data, Label) in enumerate(train_loader):
              Data = Data.to(device).float().squeeze().reshape([-1, coords*sampleLength])
              Label = Label.to(device).float().squeeze()
              optimizer.zero_grad()
              response = Net(Data).squeeze()
              loss = criterion(response, Label)
              loss.backward()
              optimizer.step()
              if batch_idx%10==0:
                      print("Loss: "+str(loss.item()))
              
      MeanAcc=0
      total_batches = 0
      total_mse = 0
      for batch_idx, (Data, Label) in enumerate(test_loader):
              Data = Data.to(device).float().squeeze().reshape([-1, coords*sampleLength])
              Label = Label.to(device).float().squeeze()

              # No sigmoid activation for regression
              response = Net(Data).squeeze()  
              
              # My own error function
              loss = my_criterion(response, Label)
              total_batches += 1
              total_mse += loss
      
      mean_mse = total_mse / total_batches
      print(f"Mean error: {mean_mse:.4f}")
      torch.save(Net.state_dict(),'lightning_regressor.pth')   
