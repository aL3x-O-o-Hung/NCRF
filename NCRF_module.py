import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self,channels,height,width,is_pe_learnable=True):
        super(PositionalEncoding,self).__init__()
        y_position=torch.arange(height).float()
        x_position=torch.arange(width).float()
        y_position,x_position=torch.meshgrid(y_position,x_position,indexing='ij')
        y_position=y_position.flatten().unsqueeze(0)
        x_position=x_position.flatten().unsqueeze(0)
        div_term=torch.exp(torch.arange(0,channels//4).float()*(-math.log(10000.0)/(channels//4))).unsqueeze(1)
        pe=torch.zeros(1,channels,height*width)
        pe[0,0::4,:]=torch.sin(y_position*div_term)
        pe[0,1::4,:]=torch.cos(y_position*div_term)
        pe[0,2::4,:]=torch.sin(x_position*div_term)
        pe[0,3::4,:]=torch.cos(x_position*div_term)
        pe=pe.view(1,channels,height,width)
        self.pe=nn.Parameter(pe,requires_grad=is_pe_learnable)
    def forward(self,x):
        return x+self.pe

class PairwisePotentialsEncoder(nn.Module):
    def __init__(self,num_classes,channels,height,width,is_pe_learnable,nearby):
        super(PairwisePotentialsEncoder,self).__init__()
        self.num_classes=num_classes
        self.nearby=nearby
        self.height=height
        self.width=width
        self.channels=channels
        self.pe=PositionalEncoding(channels,height,width,is_pe_learnable)
        self.conv=[]
        self.conv.append(ConvBlock1x1(channels,channels))
        self.conv.append(nn.Conv2d(in_channels=channels,out_channels=channels*num_classes,kernel_size=(1,1)))
        self.conv=nn.Sequential(*self.conv)
        self.potential=[]
        self.potential.append(nn.Linear(in_features=channels,out_features=1))
        self.potential=nn.Sequential(*self.potential)
    def forward(self,x):
        x=self.pe(x)
        x=self.conv(x).view(x.size(0),self.num_classes,self.channels,self.height,self.width)
        y=torch.zeros((x.size(0),self.num_classes,self.num_classes,2*self.nearby+1,2*self.nearby+1,self.height,self.width),device=x.device)

        for dx in range(-self.nearby,self.nearby+1):
            for dy in range(-self.nearby,self.nearby+1):
                if dx==0 and dy==0:
                    continue
                shifted_x=torch.roll(x,shifts=(dx,dy),dims=(-2,-1))
                diff=(x.unsqueeze(2)-shifted_x.unsqueeze(1))**2
                diff=torch.permute(diff,(0,1,2,4,5,3))
                potential=self.potential(diff)
                y[:,:,:,dx+self.nearby,dy+self.nearby,:,:]=potential[:,:,:,:,:,0]
        return y.exp()



class NCRF(nn.Module):
    def __init__(self,nearby,num_classes,num_iter=10):
        super().__init__()
        self.nearby=nearby
        self.num_classes=num_classes
        self.num_iter=num_iter
    def forward(self,unary,pair):
        q_=unary
        pair=torch.clamp(pair,min=1e-3,max=100)
        for i in range(self.num_iter):
            q=F.softmax(q_,dim=1)
            q_temp=torch.zeros_like(q_)
            for dx in range(-self.nearby,self.nearby+1):
                for dy in range(-self.nearby,self.nearby+1):
                    shifted_q=torch.roll(q,shifts=(dx,dy),dims=(-2,-1))
                    q_temp=q_temp+torch.einsum('bnhw,bcdhw->bchw',shifted_q,pair[:,:,:,dx+self.nearby,dy+self.nearby,:,:])
            q_=unary-q_temp
        return q_



