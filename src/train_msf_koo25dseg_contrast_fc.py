import argparse
from distutils.command.config import config
import logging
import os
import sys

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
import SimpleITK as sitk
import random
import numpy as np

from MSF25dseg_gif_supcon import MSFF_basic, MSFKoosNet
from dataloader_CMDA22 import CMDA2022_T1KOO_label, CMDA2022_T2KOO
from losses import PerceptualLoss, GANLoss, SoftDiceLoss, SupConLoss
from utils import poly_lr, Recorder, Plotter

def train(
    net,
    netS,
    device,
    epochs=5,
    batch_size=1,
    lr=0.001,
    dir_checkpoint='',
    dir_visualize='',
    fold=0,
):
    train_data1 = CMDA2022_T1KOO_label(mode='train', nameclass='1\n', fold=fold)
    train_data2 = CMDA2022_T1KOO_label(mode='train', nameclass='2\n', fold=fold)
    train_data3 = CMDA2022_T1KOO_label(mode='train', nameclass='3\n', fold=fold)
    train_data4 = CMDA2022_T1KOO_label(mode='train', nameclass='4\n', fold=fold)
    
    valid_data1 = CMDA2022_T1KOO_label(mode='valid', nameclass='1\n', fold=fold)
    valid_data2 = CMDA2022_T1KOO_label(mode='valid', nameclass='2\n', fold=fold)
    valid_data3 = CMDA2022_T1KOO_label(mode='valid', nameclass='3\n', fold=fold)
    valid_data4 = CMDA2022_T1KOO_label(mode='valid', nameclass='4\n', fold=fold)

    n_train = 50#len(dataset) - n_val
    n_valid = [len(valid_data1), len(valid_data2), len(valid_data3), len(valid_data4)]#len(dataset) - n_val

    train_loader1 = DataLoader(train_data1, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    train_loader2 = DataLoader(train_data2, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    train_loader3 = DataLoader(train_data3, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    train_loader4 = DataLoader(train_data4, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    valid_loader1 = DataLoader(valid_data1, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    valid_loader2 = DataLoader(valid_data2, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    valid_loader3 = DataLoader(valid_data3, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    valid_loader4 = DataLoader(valid_data4, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    iter_1 = iter(train_loader1)
    iter_2 = iter(train_loader2)
    iter_3 = iter(train_loader3)
    iter_4 = iter(train_loader4)
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {[len(train_data1), len(train_data2), len(train_data3), len(train_data4)]}
        Valid size:      {n_valid}
        Device:          {device.type}
    ''')
    
    optimizer = torch.optim.Adam(net.classifier.parameters(), lr=lr, weight_decay=1e-8)
    lr0 = poly_lr(0, epochs, lr, min_lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: poly_lr(epoch, epochs, lr, min_lr=1e-5)/lr0)
      
    recorder = Recorder(['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    plotter = Plotter(dir_visualize, keys1=['train_loss', 'valid_loss'], keys2=['train_acc', 'valid_acc'])
    total_step = 0
    best_acc = 0
    with open(os.path.join(dir_checkpoint, 'log.csv'), 'w') as f:
        f.write('epoch, loss, acc\n')


    for epoch in range(epochs):
        if epoch!=0:
            scheduler.step()
            print(epoch, optimizer.param_groups[0]['lr'])
        net.train()
        netS.train()
        train_losses = []
        train_acc = []
        valid_losses = []
        valid_acc = []
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            
            for step in range(n_train):
                
                try:
                    batch = next(iter_1)
                except:
                    iter_1 = iter(train_loader1)
                    batch = next(iter_1)
                imgsT1 = batch['imgT1']*2-1
                imgsT2 = batch['imgT2']*2-1

                b,c,d,w,h = imgsT1.shape
                imgsT1 = torch.cat([imgsT1[:,:,0:d-2], imgsT1[:,:,1:d-1], imgsT1[:,:,2:d]], dim=1)
                segsT1 = segsT1[:,:,1:d-1]
                b,c,d,w,h = imgsT2.shape
                imgsT2 = torch.cat([imgsT2[:,:,0:d-2], imgsT2[:,:,1:d-1], imgsT2[:,:,2:d]], dim=1)
                
                with torch.no_grad():
                    bs, nw, nh = 32, 224, 224 #352, 480

                    b,c,d,w,h = imgsT1.shape
                    rd = random.randint(0, d-bs-1) if d>bs else 0
                    rw = random.randint(0, w-nw-1) if w>nw else 0
                    rh = random.randint(0, h-nh-1) if h>nh else 0
                    imgs_t1p_1 = imgsT1[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                    imgs_t2p_1 = imgsT2[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                    segs_t1p_1 = segsT1[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                    
                    enc1_1 = netS.encoders(imgs_t1p_1)[0]
                    enc2_1 = netS.encoders(imgs_t2p_1)[0]
                    enc_mask = F.interpolate(segs_t1p_1, size=enc1_1.shape[-2:], mode='bilinear', align_corners=False)
                    enc1_1 = torch.cat([enc1_1, enc_mask], dim=1).permute(1,0,2,3).unsqueeze(0)
                    enc2_1 = torch.cat([enc2_1, enc_mask], dim=1).permute(1,0,2,3).unsqueeze(0)



                try:
                    batch = next(iter_2)
                except:
                    iter_2 = iter(train_loader2)
                    batch = next(iter_2)
                imgsT1 = batch['imgT1']*2-1
                imgsT2 = batch['imgT2']*2-1
                segsT1 = batch['segT1']

                b,c,d,w,h = imgsT1.shape
                imgsT1 = torch.cat([imgsT1[:,:,0:d-2], imgsT1[:,:,1:d-1], imgsT1[:,:,2:d]], dim=1)
                segsT1 = segsT1[:,:,1:d-1]
                b,c,d,w,h = imgsT2.shape
                imgsT2 = torch.cat([imgsT2[:,:,0:d-2], imgsT2[:,:,1:d-1], imgsT2[:,:,2:d]], dim=1)
                
                with torch.no_grad():
                    bs, nw, nh = 32, 224, 224 #352, 480

                    b,c,d,w,h = imgsT1.shape
                    rd = random.randint(0, d-bs-1) if d>bs else 0
                    rw = random.randint(0, w-nw-1) if w>nw else 0
                    rh = random.randint(0, h-nh-1) if h>nh else 0
                    imgs_t1p_2 = imgsT1[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                    imgs_t2p_2 = imgsT2[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                    segs_t1p_2 = segsT1[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                    
                    enc1_2 = netS.encoders(imgs_t1p_2)[0]
                    enc2_2 = netS.encoders(imgs_t2p_2)[0]
                    enc_mask = F.interpolate(segs_t1p_2, size=enc1_2.shape[-2:], mode='bilinear', align_corners=False)
                    enc1_2 = torch.cat([enc1_2, enc_mask], dim=1).permute(1,0,2,3).unsqueeze(0)
                    enc2_2 = torch.cat([enc2_2, enc_mask], dim=1).permute(1,0,2,3).unsqueeze(0)




                try:
                    batch = next(iter_3)
                except:
                    iter_3 = iter(train_loader3)
                    batch = next(iter_3)
                imgsT1 = batch['imgT1']*2-1
                imgsT2 = batch['imgT2']*2-1
                segsT1 = batch['segT1']

                b,c,d,w,h = imgsT1.shape
                imgsT1 = torch.cat([imgsT1[:,:,0:d-2], imgsT1[:,:,1:d-1], imgsT1[:,:,2:d]], dim=1)
                segsT1 = segsT1[:,:,1:d-1]
                b,c,d,w,h = imgsT2.shape
                imgsT2 = torch.cat([imgsT2[:,:,0:d-2], imgsT2[:,:,1:d-1], imgsT2[:,:,2:d]], dim=1)
                
                
                with torch.no_grad():
                    bs, nw, nh = 32, 224, 224 #352, 480

                    b,c,d,w,h = imgsT1.shape
                    rd = random.randint(0, d-bs-1) if d>bs else 0
                    rw = random.randint(0, w-nw-1) if w>nw else 0
                    rh = random.randint(0, h-nh-1) if h>nh else 0
                    imgs_t1p_3 = imgsT1[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                    imgs_t2p_3 = imgsT2[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                    segs_t1p_3 = segsT1[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                    
                    enc1_3 = netS.encoders(imgs_t1p_3)[0]
                    enc2_3 = netS.encoders(imgs_t2p_3)[0]
                    enc_mask = F.interpolate(segs_t1p_3, size=enc1_3.shape[-2:], mode='bilinear', align_corners=False)
                    enc1_3 = torch.cat([enc1_3, enc_mask], dim=1).permute(1,0,2,3).unsqueeze(0)
                    enc2_3 = torch.cat([enc2_3, enc_mask], dim=1).permute(1,0,2,3).unsqueeze(0)







                try:
                    batch = next(iter_4)
                except:
                    iter_4 = iter(train_loader4)
                    batch = next(iter_4)
                imgsT1 = batch['imgT1']*2-1
                imgsT2 = batch['imgT2']*2-1
                segsT1 = batch['segT1']

                b,c,d,w,h = imgsT1.shape
                imgsT1 = torch.cat([imgsT1[:,:,0:d-2], imgsT1[:,:,1:d-1], imgsT1[:,:,2:d]], dim=1)
                segsT1 = segsT1[:,:,1:d-1]
                b,c,d,w,h = imgsT2.shape
                imgsT2 = torch.cat([imgsT2[:,:,0:d-2], imgsT2[:,:,1:d-1], imgsT2[:,:,2:d]], dim=1)
                
                with torch.no_grad():
                    bs, nw, nh = 32, 224, 224 #352, 480

                    b,c,d,w,h = imgsT1.shape
                    rd = random.randint(0, d-bs-1) if d>bs else 0
                    rw = random.randint(0, w-nw-1) if w>nw else 0
                    rh = random.randint(0, h-nh-1) if h>nh else 0
                    imgs_t1p_4 = imgsT1[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                    imgs_t2p_4 = imgsT2[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                    segs_t1p_4 = segsT1[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                    
                    enc1_4 = netS.encoders(imgs_t1p_4)[0]
                    enc2_4 = netS.encoders(imgs_t2p_4)[0]
                    enc_mask = F.interpolate(segs_t1p_4, size=enc1_4.shape[-2:], mode='bilinear', align_corners=False)
                    enc1_4 = torch.cat([enc1_4, enc_mask], dim=1).permute(1,0,2,3).unsqueeze(0)
                    enc2_4 = torch.cat([enc2_4, enc_mask], dim=1).permute(1,0,2,3).unsqueeze(0)


                input_enc = torch.cat([
                    enc1_1, enc1_2, enc1_3, enc1_4,         
                    enc2_1, enc2_2, enc2_3, enc2_4,         
                ], dim=0)

                gt = torch.from_numpy(np.array([1,2,3,4,1,2,3,4])).to(device=device, dtype=torch.int64)-1
                pred = net(input_enc, pretrain=False)
                
                loss = nn.CrossEntropyLoss()(pred, gt) 
                acc = torch.mean((gt==torch.argmax(pred, 1)).to(dtype=torch.float32))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(**{'loss': loss.item(), 'acc': acc.item()})
                pbar.update(imgsT1.shape[0])
                train_losses.append(loss.item())
                train_acc.append(acc.item())

                total_step += 1

        

        netS.eval()
        net.eval()
        for batch in valid_loader1:
            imgsT1 = batch['imgT1']*2-1
            imgsT2 = batch['imgT2']*2-1
            segsT1 = batch['segT1']
            tumor_range = batch['tumor_range']
            d1, d2 = tumor_range

            b,c,d,w,h = imgsT1.shape
            imgsT1 = torch.cat([imgsT1[:,:,0:d-2], imgsT1[:,:,1:d-1], imgsT1[:,:,2:d]], dim=1)
            segsT1 = segsT1[:,:,1:d-1]
            b,c,d,w,h = imgsT2.shape
            imgsT2 = torch.cat([imgsT2[:,:,0:d-2], imgsT2[:,:,1:d-1], imgsT2[:,:,2:d]], dim=1)
            
            with torch.no_grad():
                bs, nw, nh = 32, 256, 256 #352, 480

                b,c,d,w,h = imgsT1.shape
                rd = (d1 + d2)//2 if d>bs else 0
                rw = 0#random.randint(0, w-nw-1) if w>nw else 0
                rh = 0#random.randint(0, h-nh-1) if h>nh else 0
                imgs_t1p_1 = imgsT1[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                imgs_t2p_1 = imgsT2[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                segs_t1p_1 = segsT1[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                
                enc1_1 = netS.encoders(imgs_t1p_1)[0]
                enc2_1 = netS.encoders(imgs_t2p_1)[0]
                enc_mask = F.interpolate(segs_t1p_1, size=enc1_1.shape[-2:], mode='bilinear', align_corners=False)
                enc1_1 = torch.cat([enc1_1, enc_mask], dim=1).permute(1,0,2,3).unsqueeze(0)
                enc2_1 = torch.cat([enc2_1, enc_mask], dim=1).permute(1,0,2,3).unsqueeze(0)

                input_enc = torch.cat([enc1_1, enc2_1], dim=0)
                pred = net(input_enc, pretrain=False)
                gt = torch.from_numpy(np.array([1,1])).to(device=device, dtype=torch.int64)-1
                
                valid_acc.append(torch.mean((gt==torch.argmax(pred, 1)).to(dtype=torch.float32)).item())
                valid_losses.append(nn.CrossEntropyLoss()(pred, gt).item())
        


        for batch in valid_loader2:
            imgsT1 = batch['imgT1']*2-1
            imgsT2 = batch['imgT2']*2-1
            segsT1 = batch['segT1']
            tumor_range = batch['tumor_range']
            d1, d2 = tumor_range

            b,c,d,w,h = imgsT1.shape
            imgsT1 = torch.cat([imgsT1[:,:,0:d-2], imgsT1[:,:,1:d-1], imgsT1[:,:,2:d]], dim=1)
            segsT1 = segsT1[:,:,1:d-1]
            b,c,d,w,h = imgsT2.shape
            imgsT2 = torch.cat([imgsT2[:,:,0:d-2], imgsT2[:,:,1:d-1], imgsT2[:,:,2:d]], dim=1)
            
            with torch.no_grad():
                bs, nw, nh = 32, 256, 256 #352, 480

                b,c,d,w,h = imgsT1.shape
                rd = (d1 + d2)//2 if d>bs else 0
                rw = 0#random.randint(0, w-nw-1) if w>nw else 0
                rh = 0#random.randint(0, h-nh-1) if h>nh else 0
                imgs_t1p_1 = imgsT1[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                imgs_t2p_1 = imgsT2[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                segs_t1p_1 = segsT1[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                
                enc1_1 = netS.encoders(imgs_t1p_1)[0]
                enc2_1 = netS.encoders(imgs_t2p_1)[0]
                enc_mask = F.interpolate(segs_t1p_1, size=enc1_1.shape[-2:], mode='bilinear', align_corners=False)
                enc1_1 = torch.cat([enc1_1, enc_mask], dim=1).permute(1,0,2,3).unsqueeze(0)
                enc2_1 = torch.cat([enc2_1, enc_mask], dim=1).permute(1,0,2,3).unsqueeze(0)

                input_enc = torch.cat([enc1_1, enc2_1], dim=0)
                pred = net(input_enc, pretrain=False)
                gt = torch.from_numpy(np.array([2,2])).to(device=device, dtype=torch.int64)-1
                
                valid_acc.append(torch.mean((gt==torch.argmax(pred, 1)).to(dtype=torch.float32)).item())
                valid_losses.append(nn.CrossEntropyLoss()(pred, gt).item())
        


        for batch in valid_loader3:
            imgsT1 = batch['imgT1']*2-1
            imgsT2 = batch['imgT2']*2-1
            segsT1 = batch['segT1']
            tumor_range = batch['tumor_range']
            d1, d2 = tumor_range

            b,c,d,w,h = imgsT1.shape
            imgsT1 = torch.cat([imgsT1[:,:,0:d-2], imgsT1[:,:,1:d-1], imgsT1[:,:,2:d]], dim=1)
            segsT1 = segsT1[:,:,1:d-1]
            b,c,d,w,h = imgsT2.shape
            imgsT2 = torch.cat([imgsT2[:,:,0:d-2], imgsT2[:,:,1:d-1], imgsT2[:,:,2:d]], dim=1)
            
            with torch.no_grad():
                bs, nw, nh = 32, 256, 256 #352, 480

                b,c,d,w,h = imgsT1.shape
                rd = (d1 + d2)//2 if d>bs else 0
                rw = 0#random.randint(0, w-nw-1) if w>nw else 0
                rh = 0#random.randint(0, h-nh-1) if h>nh else 0
                imgs_t1p_1 = imgsT1[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                imgs_t2p_1 = imgsT2[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                segs_t1p_1 = segsT1[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                
                enc1_1 = netS.encoders(imgs_t1p_1)[0]
                enc2_1 = netS.encoders(imgs_t2p_1)[0]
                enc_mask = F.interpolate(segs_t1p_1, size=enc1_1.shape[-2:], mode='bilinear', align_corners=False)
                enc1_1 = torch.cat([enc1_1, enc_mask], dim=1).permute(1,0,2,3).unsqueeze(0)
                enc2_1 = torch.cat([enc2_1, enc_mask], dim=1).permute(1,0,2,3).unsqueeze(0)

                input_enc = torch.cat([enc1_1, enc2_1], dim=0)
                pred = net(input_enc, pretrain=False)
                gt = torch.from_numpy(np.array([3,3])).to(device=device, dtype=torch.int64)-1
                
                valid_acc.append(torch.mean((gt==torch.argmax(pred, 1)).to(dtype=torch.float32)).item())
                valid_losses.append(nn.CrossEntropyLoss()(pred, gt).item())
            
        
        for batch in valid_loader4:
            imgsT1 = batch['imgT1']*2-1
            imgsT2 = batch['imgT2']*2-1
            segsT1 = batch['segT1']
            tumor_range = batch['tumor_range']
            d1, d2 = tumor_range

            b,c,d,w,h = imgsT1.shape
            imgsT1 = torch.cat([imgsT1[:,:,0:d-2], imgsT1[:,:,1:d-1], imgsT1[:,:,2:d]], dim=1)
            segsT1 = segsT1[:,:,1:d-1]
            b,c,d,w,h = imgsT2.shape
            imgsT2 = torch.cat([imgsT2[:,:,0:d-2], imgsT2[:,:,1:d-1], imgsT2[:,:,2:d]], dim=1)
           
        
            with torch.no_grad():
                bs, nw, nh = 32, 256, 256 #352, 480

                b,c,d,w,h = imgsT1.shape
                rd = (d1 + d2)//2 if d>bs else 0
                rw = 0#random.randint(0, w-nw-1) if w>nw else 0
                rh = 0#random.randint(0, h-nh-1) if h>nh else 0
                imgs_t1p_1 = imgsT1[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                imgs_t2p_1 = imgsT2[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                segs_t1p_1 = segsT1[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                
                enc1_1 = netS.encoders(imgs_t1p_1)[0]
                enc2_1 = netS.encoders(imgs_t2p_1)[0]
                enc_mask = F.interpolate(segs_t1p_1, size=enc1_1.shape[-2:], mode='bilinear', align_corners=False)
                enc1_1 = torch.cat([enc1_1, enc_mask], dim=1).permute(1,0,2,3).unsqueeze(0)
                enc2_1 = torch.cat([enc2_1, enc_mask], dim=1).permute(1,0,2,3).unsqueeze(0)

                input_enc = torch.cat([enc1_1, enc2_1], dim=0)
                pred = net(input_enc, pretrain=False)
                gt = torch.from_numpy(np.array([4,4])).to(device=device, dtype=torch.int64)-1
                
                valid_acc.append(torch.mean((gt==torch.argmax(pred, 1)).to(dtype=torch.float32)).item())
                valid_losses.append(nn.CrossEntropyLoss()(pred, gt).item())


        recorder.update({'train_loss': np.mean(train_losses), 'train_acc': np.mean(train_acc), 'valid_loss': np.mean(valid_losses), 'valid_acc': np.mean(valid_acc)})
        plotter.send(recorder.call())
        valid_acc = np.mean(valid_acc)
        if best_acc<valid_acc:
            best_acc = valid_acc
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_best.pth'))
            with open(os.path.join(dir_checkpoint, 'log.csv'), 'a+') as f:
                f.write('{}, {}, {}\n'.format(epoch+1, np.mean(valid_losses), valid_acc))
        torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_latest.pth'))
        torch.cuda.empty_cache()
                
     

def get_args():
    parser = argparse.ArgumentParser(description='Train the VGG on images and target label',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-6,
                        help='Learning rate', dest='lr')
    parser.add_argument('-c', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-m', '--load_maf', dest='load_msf', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-d', '--device', dest='device', type=str, default='cpu',
                        help='cuda or cpu')
    parser.add_argument('-s', '--save', dest='save', type=str, default='ckpt/vgg',
                        help='save ckpt')
    parser.add_argument('-v', '--visual', dest='visual', type=str, default='vis/vgg',
                        help='save visualization')
    parser.add_argument('-f', '--fold', metavar='F', type=int, default=0,
                        help='fold', dest='fold')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    dir_checkpoint = args.save
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    
    dir_visualize = args.visual
    if not os.path.exists(dir_visualize):
        os.makedirs(dir_visualize)

    device = torch.device(args.device)
    logging.info(f'Using device {device}')

    netS = MSFF_basic(3, 3, 64, 2)
    netS.to(device=device)
    netS.load_state_dict(torch.load(args.load_msf, map_location=device))
    #net.autoencoders.load_state_dict(torch.load('ckpt/msf/rec/both/ckpt_v1.pth', map_location=device))
    net = MSFKoosNet(256, 'mlp', 128, 4)
    net.to(device=device)

    if args.load:
        load_dict = torch.load(args.load, map_location=device)
        net.load_state_dict(load_dict)
        print('[*] Load model from', args.load)
    
    try:
        train(
            net=net,
            netS=netS,
            device=device,
            epochs=args.epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            dir_checkpoint=dir_checkpoint,
            dir_visualize=dir_visualize,
            fold=args.fold,
        )
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)