import torch 
import torch.nn as nn
print(torch.__version__)

class Block(nn.Module):
    def __init__(self,in_channel,out_channel,stride) :
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,4,stride,1,bias=True,padding_mode='reflect'),
            nn.InstanceNorm2d(out_channel),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self,x) :
        return self.conv(x)

class Discriminator(nn.Module) :
    def __init__(self,in_channels=3,features = [64,128,256,512]) :
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels,features[0],4,2,1,padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        
        layers = []
        
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels,feature,1 if feature == features[-1] else 2))
            in_channels = feature
        
        layers.append(nn.Conv2d(in_channels,1,4,1,1,padding_mode='reflect'))
        self.model = nn.Sequential(*layers)
    
    def forward(self,x) :
        x = self.initial(x)
        return torch.sigmoid(self.model(x))


def test_discriminator() :
    x = torch.randn(2,3,256,256)
    model = Discriminator()
    y = model(x)
    print(y.shape)
    

if __name__ == '__main__' :
    test_discriminator()