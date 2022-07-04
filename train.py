import torch
from dataLoader import HorseZebraDataset
from discriminator import Discriminator
from generator import Generator
import config
import torch.optim as optim
import torch.nn as nn
from utils import loadCheckpoint,saveCheckpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

def train(disc_h,disc_z,gen_h,gen_z,loader,optim_disc,optim_gen,L1,MSE):
    loop = tqdm(loader,leave = True)
    
    for idx,data in enumerate(loop):
        zebra = data['zebra'].to(config.DEVICE)
        horse = data['horse'].to(config.DEVICE)
        
        # discriminator training
        # with torch.cuda.amp.autocast():
        fake_h = gen_h(zebra)
        disc_h_real = disc_h(horse)
        disc_h_fake = disc_h(fake_h.detach())
        disc_h_real_loss = MSE(disc_h_real,torch.ones_like(disc_h_real))
        disc_h_fake_loss = MSE(disc_h_fake,torch.zeros_like(disc_h_fake))
        disc_h_loss = (disc_h_real_loss + disc_h_fake_loss)
        
        fake_z = gen_z(horse)
        disc_z_real = disc_z(zebra)
        disc_z_fake = disc_z(fake_z.detach())
        disc_z_real_loss = MSE(disc_z_real,torch.ones_like(disc_z_real))
        disc_z_fake_loss = MSE(disc_z_fake,torch.zeros_like(disc_z_fake))
        disc_z_loss = (disc_z_real_loss + disc_z_fake_loss)
        
        disc_loss = (disc_h_loss + disc_z_loss)/2
        
        optim_disc.zero_grad()
        disc_loss.backward()
        optim_disc.step()
        
        
        # generator training
        # with torch.cuda.amp.autocast():
        # adversarial loss
        disc_h_fake = disc_h(fake_h)
        disc_z_fake = disc_z(fake_z)
        gen_h_loss = MSE(disc_h_fake,torch.ones_like(disc_h_fake))
        gen_z_loss = MSE(disc_z_fake,torch.ones_like(disc_z_fake))
        
        # cycle loss
        
        cycle_h = gen_h(fake_z)
        cycle_z = gen_z(fake_h)
        cycle_h_loss = L1(cycle_h,horse)
        cycle_z_loss = L1(cycle_z,zebra)
        
        
        gen_loss = (gen_h_loss + gen_z_loss + (cycle_h_loss + cycle_z_loss)*(config.LAMBDA_CYCLE))
        
        # indentity loss
        if(config.LAMBDA_INDENTITY>0.01):
            identity_h = gen_h(horse)
            identity_z = gen_z(zebra)
            identity_h_loss = L1(identity_h,horse)
            identity_z_loss = L1(identity_z,zebra)
            gen_loss += (identity_h_loss + identity_z_loss)*(config.LAMBDA_INDENTITY)
            
        optim_gen.zero_grad()
        gen_loss.backward()
        optim_gen.step()
        
        
        if idx%50 == 0:
            save_image(fake_h*0.5 + 0.5,f"saved_images/horse_{idx}.png")
            save_image(fake_z*0.5 + 0.5,f"saved_images/zebra_{idx}.png")

def main():
    print(config.DEVICE)
    disc_h = Discriminator().to(config.DEVICE)
    disc_z = Discriminator().to(config.DEVICE)
    gen_h = Generator().to(config.DEVICE)
    gen_z = Generator().to(config.DEVICE)
    
    optim_disc = optim.Adam(
        list(disc_h.parameters()) + list(disc_z.parameters()),
        lr = config.LEARNING_RATE,
        betas = (0.5,0.999),
    )
    optim_gen = optim.Adam(
        list(gen_h.parameters()) + list(gen_z.parameters()),
        lr = config.LEARNING_RATE,
        betas = (0.5,0.999),
    )
    
    L1 = nn.L1Loss()
    MSE = nn.MSELoss()
    
    if config.LOAD_MODEL:
        loadCheckpoint(config.CHECKPOINT_GEN_H,gen_h,optim_gen,config.LEARNING_RATE)
        loadCheckpoint(config.CHECKPOINT_GEN_Z,gen_z,optim_gen,config.LEARNING_RATE)
        loadCheckpoint(config.CHECKPOINT_CRITIC_H,disc_h,optim_disc,config.LEARNING_RATE)
        loadCheckpoint(config.CHECKPOINT_CRITIC_Z,disc_z,optim_disc,config.LEARNING_RATE)
    
    trainDataset = HorseZebraDataset(rootHorse=config.TRAIN_DIR + "/trainA",rootZebra=config.TRAIN_DIR+"/trainB",transform=config.transforms)
    # valDataset = HorseZebraDataset(rootHorse=config.TRAIN_DIR + "/testA",rootZebra=config.TRAIN_DIR+"/testB",transform=config.transforms)
    # print(trainDataset.__len__())
    loader = DataLoader(
        trainDataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    
    for epoch in range(config.NUM_EPOCHS):
        train(disc_h=disc_h,disc_z=disc_z,gen_h=gen_h,gen_z=gen_z,loader=loader,optim_disc=optim_disc,optim_gen=optim_gen,L1=L1,MSE=MSE)
        
        if config.SAVE_MODEL:
            saveCheckpoint(gen_h,optim_gen,config.CHECKPOINT_GEN_H)
            saveCheckpoint(gen_z,optim_gen,config.CHECKPOINT_GEN_Z)
            saveCheckpoint(disc_h,optim_disc,config.CHECKPOINT_CRITIC_H)
            saveCheckpoint(disc_z,optim_disc,config.CHECKPOINT_CRITIC_Z)

if __name__ == '__main__':
    main()