import argparse
from glob import glob
import logging
import os
import sys

import torch
import SimpleITK as sitk
import numpy as np
import yaml

from MSF25dseg_gif_supcon import MSFF_basic


with open('config/data.yaml', 'r') as f:
    config = yaml.safe_load(f)


def train(
    net,
    device,
):
    root_t1 = config['preprocess']['crop']['source']
    root_t2 = config['preprocess']['crop']['target']
    save_t1 = config['MSF']['fakesource']
    save_t2 = config['MSF']['faketarget']
    img_list = glob(os.path.join(root_t1, '*ceT1.nii.gz'))
    test_list = glob(os.path.join(root_t2, '*T2.nii.gz'))

    if not os.path.exists(save_t1):
        os.makedirs(save_t1)
    if not os.path.exists(save_t2):
        os.makedirs(save_t2)

    net.eval()
    
    with torch.no_grad():
        for img_p in img_list:
            print(img_p)
            img_itk = sitk.ReadImage(img_p)
            img = sitk.GetArrayFromImage(img_itk)
            img = np.clip(img, 0, 2000)/2000.
            img = np.pad(img, [[1,1],[0,0],[0,0]])
            img_tensor = torch.from_numpy(img).to(device=device, dtype=torch.float32).unsqueeze(1)
            d = img_tensor.shape[0]
            img_tensor = torch.cat([img_tensor[0:d-2], img_tensor[1:d-1], img_tensor[2:d]], dim=1)
            
            output = net(img_tensor)
            
            output = output['out']
            output2 = output[1][:,1].cpu().numpy()/2+0.5
            out = sitk.GetImageFromArray(output2)
            out.CopyInformation(img_itk)
            sitk.WriteImage(out, os.path.join(save_t1, os.path.basename(img_p)))
        
        for img_p in test_list:
            print(img_p)
            img_itk = sitk.ReadImage(img_p)
            img = sitk.GetArrayFromImage(img_itk)
            img = np.clip(img, 0, 1500)/1500.
            img = np.pad(img, [[1,1],[0,0],[0,0]])
            img_tensor = torch.from_numpy(img).to(device=device, dtype=torch.float32).unsqueeze(1)
            d = img_tensor.shape[0]
            img_tensor = torch.cat([img_tensor[0:d-2], img_tensor[1:d-1], img_tensor[2:d]], dim=1)
            
            output = net(img_tensor)

            output = output['out']
            output1 = output[0][:,1].cpu().numpy()/2+0.5
            out = sitk.GetImageFromArray(output1)
            out.CopyInformation(img_itk)
            sitk.WriteImage(out, os.path.join(save_t2, os.path.basename(img_p)))


def get_args():
    parser = argparse.ArgumentParser(description='Train the VGG on images and target label',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-d', '--device', dest='device', type=str, default='cpu',
                        help='cuda or cpu')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    device = torch.device(args.device)
    logging.info(f'Using device {device}')

    net = MSFF_basic(3,3,64,2)
    net.to(device=device)

    if args.load:
        load_dict = torch.load(args.load, map_location=device)
        net.load_state_dict(load_dict)
        print('[*] Load model from', args.load)

    try:
        train(
            net=net,
            device=device,
        )
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)