


class ConvBlock(nn.Module):
    def __init__(self,input_channels,output_channels,max_pool,return_single=False):
        super(ConvBlock,self).__init__()
        self.max_pool=max_pool
        self.conv=[]
        self.conv.append(nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1))
        self.conv.append(nn.InstanceNorm2d(output_channels))
        self.conv.append(nn.LeakyReLU())
        self.conv.append(nn.Conv2d(in_channels=output_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1))
        self.conv.append(nn.InstanceNorm2d(output_channels))
        self.conv.append(nn.LeakyReLU())
        self.return_single=return_single
        if max_pool:
            self.pool=nn.MaxPool2d(2,stride=2,dilation=(1,1))
        self.conv=nn.Sequential(*self.conv)

    def forward(self,x):
        x=self.conv(x)
        b=x
        if self.max_pool:
            x=self.pool(x)
        if self.return_single:
            return x
        else:
            return x,b





class DeconvBlock(nn.Module):
    def __init__(self,input_channels,output_channels,intermediate_channels=-1):
        super(DeconvBlock,self).__init__()
        input_channels=int(input_channels)
        output_channels=int(output_channels)
        if intermediate_channels<0:
            intermediate_channels=output_channels*2
        else:
            intermediate_channels=input_channels
        self.upconv=[]
        self.upconv.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.upconv.append(nn.Conv2d(in_channels=input_channels,out_channels=intermediate_channels//2,kernel_size=3,stride=1,padding=1))
        self.conv=ConvBlock(intermediate_channels,output_channels,False)
        self.upconv=nn.Sequential(*self.upconv)

    def forward(self,x,b):
        x=self.upconv(x)
        x=torch.cat((x,b),dim=1)
        x,_=self.conv(x)
        return x

class Encoder(nn.Module):
    def __init__(self,input_channels,num_layers,base_num):
        super(Encoder,self).__init__()
        self.conv=[]
        self.num_layers=num_layers
        for i in range(num_layers):
            if i==0:
                self.conv.append(ConvBlock(input_channels,base_num,True))
            else:
                self.conv.append(ConvBlock(base_num*(2**(i-1)),base_num*(2**i),(i!=num_layers-1)))
        self.conv=nn.Sequential(*self.conv)

    def forward(self,x):
        b=[]
        for i in range(self.num_layers):
            x,block=self.conv[i](x)
            b.append(block)
        b=b[:-1]
        b=b[::-1]
        return x,b

class Decoder(nn.Module):
    def __init__(self,num_classes,num_layers,base_num):
        super(Decoder,self).__init__()
        self.include_last=include_last
        self.conv=[]
        self.num_layers=num_layers
        for i in range(num_layers-1,0,-1):
            self.conv.append(DeconvBlock(base_num*(2**i),base_num*(2**(i-1))))
        if include_last:
            self.conv.append(nn.Conv2d(in_channels=base_num,out_channels=num_classes,kernel_size=1,stride=1,padding=0))
        self.conv=nn.Sequential(*self.conv)
    def forward(self,x,b):
        for i in range(self.num_layers-1):
            x=self.conv[i](x,b[i])
        return x


class NCRFUNet(nn.Module):
    def __init__(self,input_channels,num_classes,num_layers,base_num=64,height=128,width=128,is_pe_learnable=True,nearby=1,num_iter=10):
        super(NCRFUNet,self).__init__()
        self.encoder=Encoder(input_channels,num_layers,base_num)
        self.num_classes=num_classes
        self.decoder=Decoder(num_classes,num_layers,base_num)
        self.unary=nn.Conv2d(kernel_size=(1,1),in_channels=base_num,out_channels=num_classes)
        self.pair_encoder=PairwisePotentialsEncoder(num_classes,base_num,height,width,is_pe_learnable,nearby)
        self.ncrf=NCRF(nearby,num_classes,num_iter)

    def forward(self,x):
        x,b=self.encoder(x)
        x=self.decoder(x,b)
        y=self.pair_encoder(x)
        x=self.unary(x)
        x=self.ncrf(x,y)
        return x