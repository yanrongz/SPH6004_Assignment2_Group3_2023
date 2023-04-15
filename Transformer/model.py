# Author: Liang Jingyu
import torch
from torch.nn import Transformer, TransformerEncoderLayer, TransformerDecoderLayer
from torch import nn

class attention(nn.Module):
    def __init__(self, input_size):
        super(attention,self).__init__()
        self.encoder = TransformerEncoderLayer(d_model=input_size,nhead=1,batch_first=True,dim_feedforward=512)
        self.decoder = TransformerDecoderLayer(d_model=input_size,nhead=1,batch_first=True,dim_feedforward=512)
        self.layer = Transformer(d_model=input_size,nhead=1,dim_feedforward=512,batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(input_size,20),
            nn.Linear(20,30),
            nn.Linear(30,15),
            nn.Linear(15,5),
            nn.Linear(5,1)
        )
        self.loss = RMSELoss
        for param in self.parameters():
            param.data = param.data.to(torch.float64)
    def forward(self, X):
        encoded = self.encoder(X)
        out = self.decoder(X,encoded)
        out = self.fc(out)
        return out
    
#Error Metrics: RMSE, MAE, R2-Score using torch function
#Model Trainning in CUDA(GPU) so cannot use sklearn metric libraries
def RMSELoss(pred,y):
    return torch.sqrt(torch.mean((pred-y)**2))

def MAE(outputs, labels):
    return torch.mean(torch.abs(outputs - labels))

def r_2(outputs, labels):
    ss_res = torch.sum((labels - outputs)**2)
    ss_tot = torch.sum((labels - torch.mean(labels))**2)
    res = ss_res/ss_tot
    return 1-res
