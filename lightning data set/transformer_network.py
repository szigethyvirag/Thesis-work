import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import math
import geopy
import geopy.distance

# Model: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1

class LightningDataSet(torch.utils.data.Dataset):
    def __init__(self,sampleLength = 6, IsTest=False):

        self.Series = np.transpose(np.load('series_array.npy'),[0,2,1])
        self.length=int(self.Series.shape[0])
        self.IsTest=IsTest
        self.splitPoint = int(np.floor(self.length * 0.8)) # 255
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
sampleLength = 96

train_loader = torch.utils.data.DataLoader(LightningDataSet(sampleLength=sampleLength),batch_size=batch_size,drop_last=True) 
test_loader = torch.utils.data.DataLoader(LightningDataSet( sampleLength=sampleLength, IsTest=True),batch_size=batch_size,drop_last=True)


class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    # Constructor
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        dropout_p,
        num_layers,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.transformer_encoder_layer = nn.TransformerEncoderLayer( 
            d_model=dim_model, 
            nhead=num_heads, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer = self.transformer_encoder_layer, 
            num_layers = num_layers
            )
        self.out = nn.Linear(dim_model*2, num_tokens)
        
    def forward(self, src_):
        src = src_ 
        transformer_out = self.transformer_encoder(src)
        flattened_out = torch.flatten(transformer_out, start_dim=1)
        out = self.out(flattened_out) 
        return out
      

# the order of the elements ('words')
# Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# max_len determines how far the position can have an effect on a token (window)
class PositionalEncoding(nn.Module): 
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()

        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model) 
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) 
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) 
        
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term) 

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
        
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

Net = Transformer(num_tokens=coords, dim_model=sampleLength, num_heads=3, dropout_p=0.1, num_layers=4).to(device) 

Net.train()
epochs = 1000

optimizer = torch.optim.Adam(Net.parameters()) 
criterion = nn.MSELoss()
for e in range(epochs):  
      print("Epoch: " + str(e + 1))
      for batch_idx, (Data, Label) in enumerate(train_loader):
              Data = Data.to(device).float().squeeze()
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
              Data = Data.to(device).float().squeeze()
              Label = Label.to(device).float().squeeze()
              # No sigmoid activation for regression
              response = Net(Data).squeeze()  
              # My own error function
              loss = my_criterion(response, Label)
              total_batches += 1
              total_mse += loss
      
      mean_mse = total_mse / total_batches
      print(f"Mean error: {mean_mse:.4f}")
      torch.save(Net.state_dict(),'lightning_transformer.pth')   
