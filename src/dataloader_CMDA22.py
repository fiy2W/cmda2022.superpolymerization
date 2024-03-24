import os
import numpy as np
import random
import SimpleITK as sitk
from sklearn.model_selection import KFold
from glob import glob
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


class CMDA2022_T1KOO_label(Dataset):
    def __init__(self, mode='train', nameclass='1\n', fold=-1):
        self.mode = mode
        
        self.root_T1 = config['preprocess']['crop']['source']
        self.root_T2 = config['MSF']['fakesource']
        self.file_X = []
        with open(config['data']['train']['infos'], 'r') as f:
            strs = f.readlines()
            for line in strs[1:]:
                raw = line.split(',')
                name = raw[0].replace('"', '')
                label = raw[3].replace('"', '')
                if label==nameclass:
                    self.file_X.append(name)
                elif nameclass=='all':
                    self.file_X.append(name)

        if fold!=-1:
            kf = KFold(n_splits=5)
            ifold = 0
            for train_index, test_index in kf.split(self.file_X):
                X_train, X_test = [self.file_X[i] for i in train_index], [self.file_X[i] for i in test_index]
                if ifold==fold:
                    break
                ifold+=1
                
            if self.mode=='train':
                self.file_X = X_train
            elif self.mode=='valid':
                self.file_X = X_test
        
    def preprocess(self, x, k=[0,0,0], axis=[0,1,2]):
        d, w, h = x.shape
        if d<34:
            rd = (34-d)//2
            x = np.pad(x,[[rd,34-d-rd],[0,0],[0,0]])
        
        if self.mode!='valid':
            x = np.transpose(x, axes=axis)
            if k[0]==1:
                x = x[:, :, ::-1]
            if k[1]==1:
                x = x[:, ::-1, :]
            if k[2]==1:
                x = x[::-1, :, :]
            x = x.copy()

        return x
    
    def tumor_range(self, mask):
        ds = np.sum(mask, axis=(1,2))
        di = np.argwhere(ds>0)
        d1 = np.min(di)
        d2 = np.max(di)
        return d1, d2

    def __getitem__(self, index):
        x_idx = self.file_X[index]
        imgsT1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root_T1, x_idx+'_ceT1.nii.gz')))
        imgsT2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root_T2, x_idx+'_ceT1.nii.gz')))
        segsT1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root_T1, x_idx+'_Label.nii.gz')))

        d1, d2 = self.tumor_range(segsT1)

        imgsT1 = np.clip(imgsT1, 0, 2000)/2000.

        k = [random.randint(0,1), 0, 0]
        axis = [1,2]
        axis = [0] + axis
        imgsT1 = self.preprocess(imgsT1, k, axis)
        imgsT2 = self.preprocess(imgsT2, k, axis)
        segsT1 = self.preprocess(segsT1, k, axis)

        imgsT1 = np.expand_dims(imgsT1, axis=0)
        imgsT2 = np.expand_dims(imgsT2, axis=0)
        segsT1 = np.expand_dims(segsT1, axis=0)

        return {'imgT1': torch.from_numpy(imgsT1),
                'imgT2': torch.from_numpy(imgsT2),
                'segT1': torch.from_numpy(segsT1),
                'tumor_range': [d1, d2],
                'name': os.path.basename(x_idx).split('.')[0]}

    def __len__(self):
        return len(self.file_X)


class CMDA2022_T2KOO(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        
        self.root_T2 = config['preprocess']['crop']['target']
        self.root_T1 = config['MSF']['faketarget']
        self.root_seg = config['nnunet']['train_t2_seg']
        self.file_X = [os.path.basename(i).split('_hrT2')[0] for i in glob(os.path.join(self.root_T2, '*.nii.gz'))]
        
    def preprocess(self, x, k=[0,0,0], axis=[0,1,2]):
        d, w, h = x.shape
        if d<34:
            rd = (34-d)//2
            x = np.pad(x,[[rd,34-d-rd],[0,0],[0,0]])
        
        if self.mode!='valid':
            x = np.transpose(x, axes=axis)
            if k[0]==1:
                x = x[:, :, ::-1]
            if k[1]==1:
                x = x[:, ::-1, :]
            if k[2]==1:
                x = x[::-1, :, :]
            x = x.copy()

        return x
    
    def tumor_range(self, mask):
        ds = np.sum(mask, axis=(1,2))
        di = np.argwhere(ds>0)
        d1 = np.min(di)
        d2 = np.max(di)
        return d1, d2

    def __getitem__(self, index):
        x_idx = self.file_X[index]
        no_f = np.int32(x_idx.split('_')[-1])
        if '2022' in os.path.basename(x_idx):
            no_f += 500
        imgsT1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root_T1, x_idx+'_hrT2.nii.gz')))
        imgsT2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root_T2, x_idx+'_hrT2.nii.gz')))
        segsT1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root_seg, 'cmda_{:0>4d}.nii.gz'.format(no_f))))

        d1, d2 = self.tumor_range(segsT1)
        
        imgsT2 = np.clip(imgsT2, 0, 1500)/1500.

        k = [random.randint(0,1), 0, 0]
        axis = [1,2]
        axis = [0] + axis
        imgsT1 = self.preprocess(imgsT1, k, axis)
        imgsT2 = self.preprocess(imgsT2, k, axis)
        segsT1 = self.preprocess(segsT1, k, axis)

        imgsT1 = np.expand_dims(imgsT1, axis=0)
        imgsT2 = np.expand_dims(imgsT2, axis=0)
        segsT1 = np.expand_dims(segsT1, axis=0)

        return {'imgT1': torch.from_numpy(imgsT1),
                'imgT2': torch.from_numpy(imgsT2),
                'segT1': torch.from_numpy(segsT1),
                'tumor_range': [d1, d2],
                'name': os.path.basename(x_idx).split('.')[0]}

    def __len__(self):
        return len(self.file_X)
