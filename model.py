import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

class NNpredictor(nn.Module):
    def __init__(self,actv='tanh'):
        actv_dict={"no":nn.Identity,"sigmoid":nn.Sigmoid,"tanh":nn.Tanh}
        actv=actv_dict[actv]
        self.model=nn.Sequential(nn.Linear(7,1),actv())
    def forward(self,x):
        return self.model(x)

class experiment(pl.LightningModule):
    def __init__(self,model,lr=1e-3):
        self.model=model
        self.lr=lr
    def forward(self,x):
        return self.model(x)
    def training_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self.model(x)
        loss=F.mse_loss(y_hat,y)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

   

