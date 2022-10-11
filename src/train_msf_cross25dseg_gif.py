import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np

from MSF25dseg_gif_supcon import MSFF_basic, NLayerDiscriminator
from dataloader_CMDA22 import CMDA2022_T1T2cross
from losses import PerceptualLoss, GANLoss, SoftDiceLoss
from utils import poly_lr, Recorder, Plotter


def train(
    net,
    netD1,
    netD2,
    device,
    epochs=5,
    batch_size=1,
    lr=0.001,
    dir_checkpoint='',
    dir_visualize='',
):

    train_data = CMDA2022_T1T2cross(mode='train')
    n_train = len(train_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Device:          {device.type}
    ''')

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizerD = torch.optim.Adam([{'params': netD1.parameters()}, {'params': netD2.parameters()}], lr=lr)
    lr0 = poly_lr(0, epochs, lr, min_lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: poly_lr(epoch, epochs, lr, min_lr=1e-5)/lr0)
    schedulerD = torch.optim.lr_scheduler.LambdaLR(
        optimizerD,
        lr_lambda=lambda epoch: poly_lr(epoch, epochs, lr, min_lr=1e-5)/lr0)
    perceptual = PerceptualLoss().to(device=device)
    gan = GANLoss('lsgan').to(device=device)
        
    recorder = Recorder(['loss_rec', 'loss_per', 'loss_cyc', 'loss_d', 'loss_g', 'loss_seg'])
    plotter = Plotter(dir_visualize, keys1=['loss_rec', 'loss_cyc'], keys2=['loss_per', 'loss_seg', 'loss_d', 'loss_g'])
    
    total_step = 0
    with open(os.path.join(dir_checkpoint, 'log.csv'), 'w') as f:
        f.write('epoch, loss\n')

    for epoch in range(epochs):
        if epoch!=0:
            scheduler.step()
            schedulerD.step()
            print(epoch, optimizer.param_groups[0]['lr'])
        net.train()
        netD1.train()
        netD2.train()
        step = 1
        loss_record = {
            'rec': [], 'cyc': [], 'per': [], 'd': [], 'g': [], 'seg': []
        }
        with tqdm(total=n_train*step, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            #loader_T2 = iter(train_loader_T2)
            for batch in train_loader:
                imgsT1 = batch['imgT1']*2-1
                imgsT2 = batch['imgT2']*2-1
                segsT1 = batch['segT1']
                gifsT1 = batch['gifT1']
                name = batch['name'][0]

                b,c,d,w,h = imgsT1.shape
                imgsT1 = torch.cat([imgsT1[:,:,0:d-2], imgsT1[:,:,1:d-1], imgsT1[:,:,2:d]], dim=1)
                segsT1 = segsT1[:,:,1:d-1]
                gifsT1 = gifsT1[:,:,1:d-1]
                b,c,d,w,h = imgsT2.shape
                imgsT2 = torch.cat([imgsT2[:,:,0:d-2], imgsT2[:,:,1:d-1], imgsT2[:,:,2:d]], dim=1)

                for _ in range(step):
                    bs, nw, nh = 1, 256, 256 #352, 480

                    b,c,d,w,h = imgsT1.shape
                    rd = random.randint(0, d-bs-1) if d>bs else 0
                    rw = random.randint(0, w-nw-1) if w>nw else 0
                    rh = random.randint(0, h-nh-1) if h>nh else 0
                    imgs_t1p = imgsT1[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                    imgs_t2p = imgsT2[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                    segs_t1p = segsT1[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)
                    gifs_t1p = gifsT1[:,:,rd:rd+bs,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)

                    segs_t1p_onehot = F.one_hot(segs_t1p.to(dtype=torch.int64), num_classes=3)[:,0].permute(0,3,1,2).to(dtype=torch.float32)
                    gifs_t1p_onehot = F.one_hot(gifs_t1p.to(dtype=torch.int64), num_classes=209)[:,0].permute(0,3,1,2).to(dtype=torch.float32)
                    
                    # dis
                    with torch.no_grad():
                        output1 = net(imgs_t1p)
                        output2 = net(imgs_t2p)
                    pred_real1 = netD1(imgs_t1p)
                    pred_fake1 = netD1(output2['out'][0].detach())
                    pred_real2 = netD2(imgs_t2p)
                    pred_fake2 = netD2(output1['out'][1].detach())
                    loss_d = gan(pred_real1, True) + gan(pred_fake1, False) + gan(pred_real2, True) + gan(pred_fake2, False)
                    optimizerD.zero_grad()
                    loss_d.backward()
                    optimizerD.step()
                    
                    # gen
                    output1 = net(imgs_t1p)
                    output1_cyc = net(output1['out'][1])
                    output2 = net(imgs_t2p)
                    output2_cyc = net(output2['out'][0])

                    loss_rec = nn.SmoothL1Loss()(output1['out'][0], imgs_t1p) + nn.SmoothL1Loss()(output2['out'][1], imgs_t2p)
                    loss_cyc = nn.SmoothL1Loss()(output1_cyc['out'][0], imgs_t1p) + nn.SmoothL1Loss()(output2_cyc['out'][1], imgs_t2p)
                    loss_per = perceptual(output1['out'][0][:]/2.+0.5, imgs_t1p[:]/2.+0.5) + \
                        perceptual(output2['out'][1][:]/2.+0.5, imgs_t2p[:]/2.+0.5)
                    loss_seg = SoftDiceLoss(apply_nonlin=nn.Softmax(dim=1))(output1['mask'], segs_t1p_onehot) + \
                        SoftDiceLoss(apply_nonlin=nn.Softmax(dim=1))(output1_cyc['mask'], segs_t1p_onehot) + \
                        nn.CrossEntropyLoss()(output1['mask'], segs_t1p[:,0].to(dtype=torch.int64)) + \
                        nn.CrossEntropyLoss()(output1_cyc['mask'], segs_t1p[:,0].to(dtype=torch.int64)) + \
                        SoftDiceLoss(apply_nonlin=nn.Softmax(dim=1))(output1['gif'], gifs_t1p_onehot) + \
                        SoftDiceLoss(apply_nonlin=nn.Softmax(dim=1))(output1_cyc['gif'], gifs_t1p_onehot) + \
                        nn.CrossEntropyLoss()(output1['gif'], gifs_t1p[:,0].to(dtype=torch.int64)) + \
                        nn.CrossEntropyLoss()(output1_cyc['gif'], gifs_t1p[:,0].to(dtype=torch.int64))
                        
                    loss_adv = gan(netD1(output2['out'][0]), True) + gan(netD2(output1['out'][1]), True)
                    
                    loss = loss_rec*10 + loss_cyc*10 + loss_adv + loss_seg + loss_per*0.01

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix(**{'rec': loss_rec.item(), 'cyc': loss_cyc.item(), 'dis': loss_d.item(), 'gen': loss_adv.item(), 'seg': loss_seg.item(), 'per': loss_per.item()})
                    pbar.update(imgsT1.shape[0])
                    loss_record['rec'].append(loss_rec.item())
                    loss_record['cyc'].append(loss_cyc.item())
                    loss_record['per'].append(loss_per.item())
                    loss_record['d'].append(loss_d.item())
                    loss_record['g'].append(loss_adv.item())
                    loss_record['seg'].append(loss_seg.item())

                    if (total_step % 64) == 0:
                        outseg1 = torch.argmax(output1['mask'], dim=1, keepdim=True)-1
                        outseg1_cyc = torch.argmax(output1_cyc['mask'], dim=1, keepdim=True)-1
                        outseg2 = torch.argmax(output2['mask'], dim=1, keepdim=True)-1
                        outseg2_cyc = torch.argmax(output2_cyc['mask'], dim=1, keepdim=True)-1
                        outgif1 = torch.argmax(output1['gif'], dim=1, keepdim=True)/208/2-0.5
                        outgif1_cyc = torch.argmax(output1_cyc['gif'], dim=1, keepdim=True)/208/2-0.5
                        outgif2 = torch.argmax(output2['gif'], dim=1, keepdim=True)/208/2-0.5
                        outgif2_cyc = torch.argmax(output2_cyc['gif'], dim=1, keepdim=True)/208/2-0.5

                        save_img([
                            [imgs_t1p[i:i+1,1:2] for i in range(0, imgs_t1p.shape[0], 1)]+[imgs_t2p[i:i+1,1:2] for i in range(0, imgs_t1p.shape[0], 1)],
                            [output1['out'][0][i:i+1,1:2] for i in range(0, imgs_t1p.shape[0], 1)]+[output2['out'][0][i:i+1,1:2] for i in range(0, imgs_t1p.shape[0], 1)],
                            [output1['out'][1][i:i+1,1:2] for i in range(0, imgs_t1p.shape[0], 1)]+[output2['out'][1][i:i+1,1:2] for i in range(0, imgs_t1p.shape[0], 1)],
                            [output1_cyc['out'][0][i:i+1,1:2] for i in range(0, imgs_t1p.shape[0], 1)]+[output2_cyc['out'][0][i:i+1,1:2] for i in range(0, imgs_t1p.shape[0], 1)],
                            [output1_cyc['out'][1][i:i+1,1:2] for i in range(0, imgs_t1p.shape[0], 1)]+[output2_cyc['out'][1][i:i+1,1:2] for i in range(0, imgs_t1p.shape[0], 1)],
                            [segs_t1p[i:i+1]-1 for i in range(0, imgs_t1p.shape[0], 1)]+[segs_t1p[i:i+1]-1 for i in range(0, imgs_t1p.shape[0], 1)],
                            [outseg1[i:i+1] for i in range(0, imgs_t1p.shape[0], 1)]+[outseg2[i:i+1] for i in range(0, imgs_t1p.shape[0], 1)],
                            [outseg1_cyc[i:i+1] for i in range(0, imgs_t1p.shape[0], 1)]+[outseg2_cyc[i:i+1] for i in range(0, imgs_t1p.shape[0], 1)],
                            [gifs_t1p[i:i+1]/208/2-0.5 for i in range(0, imgs_t1p.shape[0], 1)]+[torch.zeros_like(gifs_t1p[i:i+1]) for i in range(0, imgs_t1p.shape[0], 1)],
                            [outgif1[i:i+1] for i in range(0, imgs_t1p.shape[0], 1)]+[outgif2[i:i+1] for i in range(0, imgs_t1p.shape[0], 1)],
                            [outgif1_cyc[i:i+1] for i in range(0, imgs_t1p.shape[0], 1)]+[outgif2_cyc[i:i+1] for i in range(0, imgs_t1p.shape[0], 1)],
                            ], epoch, dir_visualize)

                    total_step += 1
        
        recorder.update({
            'loss_rec': np.mean(loss_record['rec']),
            'loss_per': np.mean(loss_record['per']),
            'loss_cyc': np.mean(loss_record['cyc']),
            'loss_d': np.mean(loss_record['d']),
            'loss_g': np.mean(loss_record['g']),
            'loss_seg': np.mean(loss_record['seg'])})
        plotter.send(recorder.call())
        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_{}.pth'.format(epoch+1)))
        torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_latest.pth'))
        torch.cuda.empty_cache()
                
                
def save_img(print_list, index, results_dir):
    # pdb.set_trace()
    nrow = len(print_list[0])
    cols = []
    for row in print_list:
        cols.append(torch.cat(row, dim=3))
    img = torch.cat(cols, dim=2)
    
    directory = os.path.join(results_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, '{:04d}'.format(index) + '.jpg')
    img = img.permute(1,0,2,3).contiguous()
    img = torch.clamp(img, -1, 1)
    vutils.save_image(img.view(1,img.size(0),-1,img.size(3)).data, path, nrow=nrow, padding=0, normalize=True)


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
    parser.add_argument('-d', '--device', dest='device', type=str, default='cpu',
                        help='cuda or cpu')
    parser.add_argument('-s', '--save', dest='save', type=str, default='ckpt/vgg',
                        help='save ckpt')
    parser.add_argument('-v', '--visual', dest='visual', type=str, default='vis/vgg',
                        help='save visualization')

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

    net = MSFF_basic(3, 3, 64, 2)
    net.to(device=device)

    if args.load:
        load_dict = torch.load(args.load, map_location=device)
        net.load_state_dict(load_dict)
        print('[*] Load model from', args.load)

    netD1 = NLayerDiscriminator(input_nc=3, n_class=1, ndf=64, n_layers=6)
    netD1.to(device=device)
    netD2 = NLayerDiscriminator(input_nc=3, n_class=1, ndf=64, n_layers=6)
    netD2.to(device=device)
    
    try:
        train(
            net=net,
            netD1=netD1,
            netD2=netD2,
            device=device,
            epochs=args.epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            dir_checkpoint=dir_checkpoint,
            dir_visualize=dir_visualize,
        )
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)