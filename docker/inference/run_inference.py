#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import SimpleITK as sitk
import shutil
from scipy.ndimage import label

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunet.inference.predict import predict_from_folder

from src.preprocess import resampleDCM, histMatch
from src.MSF25dseg_gif_supcon import MSFF_basic, MSFKoosNet


def preprocess(x, k=[0,0,0], axis=[0,1,2]):
    d, w, h = x.shape
    if d<34:
        rd = (34-d)//2
        x = np.pad(x,[[rd,34-d-rd],[0,0],[0,0]])

    return x

def tumor_range(mask):
    try:
        ds = np.sum(mask, axis=(1,2))
        di = np.argwhere(ds>0)
        d1 = np.min(di)
        d2 = np.max(di)
    except:
        d1 = 0
        d2 = mask.shape[0]
    return d1, d2


tmp_dir = '/myhome/'
tmp_in_dir = os.path.join(tmp_dir, 'Task701_CMDA1/imagesTs')
tmp_out_dir = os.path.join(tmp_dir, 'output_701')
if not os.path.exists(tmp_in_dir):
    os.makedirs(tmp_in_dir)


input_dir = '/input/'
output_dir = '/output/'
path_img = os.path.join(input_dir, '{}_hrT2.nii.gz')
path_pred = os.path.join(output_dir, '{}_Label.nii.gz')

list_case = [k.split('_hrT2')[0] for k in os.listdir(input_dir)]

with open(os.path.join(output_dir, 'predictions.csv'), 'w') as f:
    f.write('"case","class"\n')

device = torch.device('cuda:0')

netS = MSFF_basic(3, 3, 64, 2)
netS.to(device=device)
netS.load_state_dict(torch.load('ckpt/cm/ckpt_1000.pth', map_location=device))

net = [
    MSFKoosNet(256, 'mlp', 128, 4),
    MSFKoosNet(256, 'mlp', 128, 4),
    MSFKoosNet(256, 'mlp', 128, 4),
    MSFKoosNet(256, 'mlp', 128, 4),
    MSFKoosNet(256, 'mlp', 128, 4),
]
for i in range(5):
    net[i].to(device=device)

net[0].load_state_dict(torch.load('ckpt/koos/f0/ckpt_11.pth', map_location=device))
net[1].load_state_dict(torch.load('ckpt/koos/f1/ckpt_12.pth', map_location=device))
net[2].load_state_dict(torch.load('ckpt/koos/f2/ckpt_6.pth', map_location=device))
net[3].load_state_dict(torch.load('ckpt/koos/f3/ckpt_9.pth', map_location=device))
net[4].load_state_dict(torch.load('ckpt/koos/f4/ckpt_8.pth', map_location=device))

for i in range(5):
    net[i].eval()
netS.eval()

for case in list_case:
    print(case)
    t2_img = sitk.ReadImage(path_img.format(case))

    ##
    # your logic here. Below we do binary thresholding as a demo
    ##
    """ preprocess """
    ### Step 1: resampling
    spacing = [0.4102, 0.4102, 1]
    t2_img_resample, _, _ = resampleDCM(t2_img, new_spacing=spacing, is_label=False)

    ### Step 2: histogram matching
    tgt_t2_img_atlas = sitk.ReadImage('atlas/crossmoda2021_ldn_106_hrT2.nii.gz')
    t2_img_hm = histMatch(t2_img_resample, tgt_t2_img_atlas)

    ### Step 3: crop fix patch
    t2_img_hm_arr = sitk.GetArrayFromImage(t2_img_hm)
    if t2_img_hm_arr.shape[0] <= 80:
        sd, ed = 0, 80
    else:
        sd = np.clip(np.int64(t2_img_hm_arr.shape[0]*0.375)-40, 0, None)
        ed = sd + 80
    t2_img_crop_arr = np.clip(t2_img_hm_arr[sd:ed, 200:456, 128:384], 0, 1500)/1500
    t2_img_crop = sitk.GetImageFromArray(t2_img_crop_arr)
    t2_img_crop.SetSpacing(t2_img_hm.GetSpacing())
    sitk.WriteImage(t2_img_crop, os.path.join(tmp_in_dir, 'cmda_0001_0000.nii.gz'))

    """ segmentation """
    if os.path.exists(tmp_out_dir):
        shutil.rmtree(tmp_out_dir)
    os.makedirs(tmp_out_dir)

    # nnunet
    predict_from_folder(model='ckpt/nnUNet/3d_fullres/Task701_CMDA1/nnUNetTrainerV2__nnUNetPlansv2.1', input_folder=tmp_in_dir, output_folder=tmp_out_dir, folds=None, save_npz=False, num_threads_preprocessing=2,
                        num_threads_nifti_save=2, lowres_segmentations=None, part_id=0, num_parts=1, tta=True,
                        overwrite_existing=False, mode='normal', overwrite_all_in_gpu=None,
                        mixed_precision=True,
                        step_size=0.5, checkpoint_name='model_best')

    # post-process
    pred = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(tmp_out_dir, 'cmda_0001.nii.gz')))
    post_pred_vs = np.zeros_like(pred)
    post_pred_co = np.zeros_like(pred)
    post_pred_vs[pred==1] = 1
    post_pred_co[pred==2] = 1

    s = [[[1,1,1],[1,1,1],[1,1,1]],
         [[1,1,1],[1,1,1],[1,1,1]],
         [[1,1,1],[1,1,1],[1,1,1]]]
    label_arr1, num_arr1 = label(post_pred_vs, structure=s)
    label_arr2, num_arr2 = label(post_pred_co, structure=s)
    
    if num_arr1>1:
        pred_cc = np.zeros_like(pred)
        pred_cc[pred==2] = 2
        d, w, h = pred.shape
        dl = np.linspace(0,d-1,d)
        wl = np.linspace(0,w-1,w)
        hl = np.linspace(0,h-1,h)
        hl, dl, wl = np.meshgrid(hl, dl, wl)
        coc_d = []
        for coc in range(1,num_arr2+1):
            coc_d.append(np.mean(dl[label_arr2==coc]))
        
        coc_d_range = np.max((np.max(coc_d) - np.min(coc_d), 20))
        selectvs = 0
        
        max_area = 0
        max_id = -1
        for vs in range(1,num_arr1+1):
            vs_d = np.mean(dl[label_arr1==vs])
            if np.abs(vs_d-np.min(coc_d))<coc_d_range or np.abs(vs_d-np.max(coc_d))<coc_d_range:
                if np.sum(label_arr1==vs) > max_area:
                    max_area = np.sum(label_arr1==vs)
                    max_id = vs
                selectvs += 1
        if max_id != -1:
            vs = max_id
            pred_cc[label_arr1==vs] = 1
                
        pred_mask = pred_cc.copy()
    else:
        pred_mask = post_pred_vs.copy()
        pred_mask[pred==2] = 2

    pred_arr = np.zeros_like(t2_img_hm_arr)
    pred_arr[sd:ed, 200:456, 128:384] = pred_mask
    result = sitk.GetImageFromArray(pred_arr)
    result.CopyInformation(t2_img_resample)

    result = resampleDCM(result, t2_img.GetSpacing(), is_label=True, new_size=t2_img.GetSize(), new_origin=t2_img.GetOrigin())[0]
    result.CopyInformation(t2_img)

    sitk.WriteImage(result, path_pred.format(case))


    """ koos """
    pred_mask = preprocess(pred_mask)
    imgs_crop_arr = preprocess(t2_img_crop_arr)
    d1, d2 = tumor_range(pred_mask)
    d1_c = np.clip(d2-32, 0, imgs_crop_arr.shape[0]-34)
    d2_c = np.clip(d1, 0, imgs_crop_arr.shape[2]-34)
    d_c = (d1_c+d2_c)//2
    imgs_crop_arr = imgs_crop_arr[d_c:d_c+34]
    pred_mask = pred_mask[d_c:d_c+34]

    with torch.no_grad():
        imgs_crop_tensor = torch.from_numpy(imgs_crop_arr).unsqueeze(0).unsqueeze(0)
        pred_mask_tensor = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0)
        d = imgs_crop_tensor.shape[2]
        imgs_crop_tensor = torch.cat([imgs_crop_tensor[:,:,0:d-2], imgs_crop_tensor[:,:,1:d-1], imgs_crop_tensor[:,:,2:d]], dim=1).to(
            device=device, dtype=torch.float32)[0].permute(1,0,2,3)
        pred_mask_tensor = pred_mask_tensor[:,:,1:d-1].to(device=device, dtype=torch.float32)[0].permute(1,0,2,3)

        enc = netS.encoders(imgs_crop_tensor)[0]
        enc_mask = F.interpolate(pred_mask_tensor, size=enc.shape[-2:], mode='bilinear', align_corners=False)
        input_enc = torch.cat([enc, enc_mask], dim=1).permute(1,0,2,3).unsqueeze(0)

        pred = torch.cat([
            nn.Softmax(dim=1)(net[0](input_enc, pretrain=False)),
            nn.Softmax(dim=1)(net[1](input_enc, pretrain=False)),
            nn.Softmax(dim=1)(net[2](input_enc, pretrain=False)),
            nn.Softmax(dim=1)(net[3](input_enc, pretrain=False)),
            nn.Softmax(dim=1)(net[4](input_enc, pretrain=False)),], dim=0)
        
        pred = torch.argmax(pred.mean(0), dim=0).item()
        with open(os.path.join(output_dir, 'predictions.csv'), 'a+') as f:
            f.write('"{}","{}"\n'.format(case, pred+1))



