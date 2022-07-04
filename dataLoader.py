import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class HorseZebraDataset(Dataset):
    def __init__(self,rootZebra,rootHorse,transform=None) :
        super().__init__()
        self.rootZebra = rootZebra
        self.rootHorse = rootHorse
        self.transform = transform
        
        self.zebraImages = os.listdir(rootZebra)
        self.horseImages = os.listdir(rootHorse)
        self.zebraLen = len(self.zebraImages)
        self.horseLen = len(self.horseImages)
        print(self.zebraLen,self.horseLen)
        self.dataLen = max(self.zebraLen,self.horseLen)
    
    def __len__(self) :
        return self.dataLen
    
    def __getitem__(self, index):
        zebraImg = self.zebraImages[index%self.zebraLen]
        horseImg = self.horseImages[index%self.horseLen]
        
        zebraPath = os.path.join(self.rootZebra,zebraImg)
        horsePath = os.path.join(self.rootHorse,horseImg) 
        
        Zebra = np.array(Image.open(zebraPath).convert("RGB"))
        Horse = np.array(Image.open(horsePath).convert("RGB"))
        
        if self.transform:
            augumenations = self.transform(image=Zebra,image0=Horse)
            zebra = augumenations["image"]
            horse = augumenations["image0"]
        
        return {"zebra" : zebra,"horse" : horse}