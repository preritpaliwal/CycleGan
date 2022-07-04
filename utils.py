import random,torch,os,numpy as np
import config 
import copy
import torch.nn as nn

def saveCheckpoint(model,optimizer,fileName = "newCheckpoint.pth") :
    print("Saving checkpoint...")
    checkPoint = {
        "state_dict":model.state_dict(),
        "optimizer":optimizer.state_dict(),
    }
    torch.save(checkPoint,fileName)

def loadCheckpoint(checkPointFile,model,optimizer,lr) :
    print("Loading checkpoint...")
    checkPoint = torch.load(checkPointFile,map_location=config.DEVICE)
    model.load_state_dict(checkPoint["state_dict"])
    optimizer.load_state_dict(checkPoint["optimizer"])
    # it is necessary to set the learning rate again
    for param_group in optimizer.param_groups :
        param_group["lr"] = lr