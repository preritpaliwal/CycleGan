import torch
import torch.nn as nn

class convBlock(nn.Module):
    def __init__(self,in_channels,out_channels,down=True,use_activation=True,**kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,padding_mode='reflect',**kwargs)
            if down
            else nn.ConvTranspose2d(in_channels,out_channels,**kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2) if use_activation else nn.Identity(),
        )
    
    def forward(self,x) :
        return self.conv(x)

class ResidualBlock(nn.Module) :
    def __init__(self,channels) :
        super().__init__()
        self.block = nn.Sequential(
            convBlock(channels,channels,kernel_size=3,padding=1),
            convBlock(channels,channels,use_activation=False,kernel_size=3,padding=1),
        )
    
    def forward(self,x) :
        return x + self.block(x)

class Generator(nn.Module) :
    def __init__(self,in_channels=3,numFeatures = 64,numResidual = 9) :
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels,numFeatures,kernel_size=7,stride=1,padding=3,padding_mode='reflect'),
            nn.ReLU(inplace=True),
        )
        self.downBlocks = nn.ModuleList(
            [
                convBlock(numFeatures,numFeatures*2,down=True,kernel_size=3,stride=2,padding=1),
                convBlock(numFeatures*2,numFeatures*4,down=True,kernel_size=3,stride=2,padding=1),
            ]
        )
        
        self.residualBlocks = nn.Sequential(
            *[ResidualBlock(numFeatures*4) for _ in range(numResidual)]
        )
        self.upBlocks = nn.ModuleList(
            [
                convBlock(numFeatures*4,numFeatures*2,down=False,kernel_size=3,stride=2,padding=1,output_padding=1),
                convBlock(numFeatures*2,numFeatures,down=False,kernel_size=3,stride=2,padding=1,output_padding=1),
            ]
        )
        
        self.last = nn.Conv2d(numFeatures,3,kernel_size=7,stride=1,padding=3,padding_mode='reflect')
    
    def forward(self,x) :
        x = self.initial(x)
        for layer in self.downBlocks:
            x = layer(x)
        x = self.residualBlocks(x)
        for layer in self.upBlocks:
            x = layer(x)
        return torch.tanh(self.last(x))



def test_generator() :
    x = torch.randn(2,3,256,256)
    model = Generator()
    y = model(x)
    print(y.shape)

if __name__ == '__main__' :
    test_generator()