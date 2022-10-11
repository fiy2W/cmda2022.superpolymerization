import os
import numpy as np
import random
import SimpleITK as sitk
import yaml

import torch
from torch.utils.data import Dataset


with open('config/data.yaml', 'r') as f:
    config = yaml.safe_load(f)


class CMDA2022_T1T2cross(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode

        self.root_t1 = config['preprocess']['crop']['source']
        self.root_t2 = config['preprocess']['crop']['target']
        self.root_gif = config['preprocess']['crop']['GIF']

        self.file_X = [os.path.join(self.root_t1, i) for i in os.listdir(self.root_t1) if 'ceT1.nii.gz' in i]
        self.file_Y = [os.path.join(self.root_t2, i) for i in os.listdir(self.root_t2) if '.nii.gz' in i]
        
    def preprocess(self, x, k=[0,0,0], axis=[0,1,2]):
        if self.mode=='train':
            x = np.transpose(x, axes=axis)
            if k[0]==1:
                x = x[:, :, ::-1]
            if k[1]==1:
                x = x[:, ::-1, :]
            if k[2]==1:
                x = x[::-1, :, :]
            x = x.copy()

        return x

    def __getitem__(self, index):
        x_idx = self.file_X[index]
        y_idx = self.file_Y[(index+random.randint(0, len(self.file_Y))) % len(self.file_Y)]
        imgsT1 = sitk.GetArrayFromImage(sitk.ReadImage(x_idx))
        segsT1 = sitk.GetArrayFromImage(sitk.ReadImage(x_idx.replace('ceT1', 'Label')))
        gifsT1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root_gif, os.path.basename(x_idx).replace('ceT1', 'gif'))))
        imgsT2 = sitk.GetArrayFromImage(sitk.ReadImage(y_idx))
        
        imgsT1 = np.clip(imgsT1, 0, 2000)/2000.
        imgsT2 = np.clip(imgsT2, 0, 1500)/1500.

        k1 = [random.randint(0,1), 0, 0]
        k2 = [random.randint(0,1), 0, 0]
        axis = [0,1,2]
        imgsT1 = self.preprocess(imgsT1, k1, axis)
        imgsT2 = self.preprocess(imgsT2, k2, axis)
        segsT1 = self.preprocess(segsT1, k1, axis)
        gifsT1 = self.preprocess(gifsT1, k1, axis)

        imgsT1 = np.expand_dims(imgsT1, axis=0)
        imgsT2 = np.expand_dims(imgsT2, axis=0)
        segsT1 = np.expand_dims(segsT1, axis=0)
        gifsT1 = np.expand_dims(gifsT1, axis=0)

        return {'imgT1': torch.from_numpy(imgsT1),
                'imgT2': torch.from_numpy(imgsT2),
                'segT1': torch.from_numpy(segsT1),
                'gifT1': torch.from_numpy(gifsT1),
                'name': os.path.basename(x_idx).split('.')[0]}

    def __len__(self):
        return len(self.file_X)
