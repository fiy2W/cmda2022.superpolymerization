import os
import json
import SimpleITK as sitk
from glob import glob
import shutil
import numpy as np
import yaml

with open('config/data.yaml', 'r') as f:
    config = yaml.safe_load(f)


# the same as $nnUNet_raw_data_base/Task110_example, the subpath must be `Task[id]_[desciption]`.
base = os.path.join(config['nnunet']['base'], 'Task701_CMDA1')
root_t1 = config['preprocess']['crop']['source']
root_t2 = config['MSF']['fakesource']
root_test = config['preprocess']['crop']['valid']
root_at2 = config['preprocess']['crop']['target']

data_dict = {
    "name": "CMDA2022",
    "description": "", 
    "author": "",
    "reference": "",
    "licence": "",
    "release": "",
    "tensorImageSize": "3D",
    "modality": {"0": "T1T2"},
    "labels": {"0": "background", "1": "VS", "2": "cochlea"},
    "numTraining": 210, "numTest": 64,
    "training": [],
    "test": []
}

imagesTr = os.path.join(base, 'imagesTr')
labelsTr = os.path.join(base, 'labelsTr')
imagesTs = os.path.join(base, 'imagesTs')
imagesT2 = os.path.join(base, 'imagesAllT2')

if not os.path.exists(imagesTr):
    os.makedirs(imagesTr)
if not os.path.exists(labelsTr):
    os.makedirs(labelsTr)
if not os.path.exists(imagesTs):
    os.makedirs(imagesTs)
if not os.path.exists(imagesT2):
    os.makedirs(imagesT2)

train_list = glob(os.path.join(root_t1, '*ceT1.nii.gz'))
test_list = glob(os.path.join(root_test, '*.nii.gz'))
allt2_list = glob(os.path.join(root_at2, '*.nii.gz'))

for f in train_list:
    no_f = np.int32(os.path.basename(f).split('_')[-2])
    if '2022' in os.path.basename(f):
        no_f += 500
        
    data_dict["training"].append({
        "image": "./imagesTr/cmda_{:04d}.nii.gz".format(no_f),
        "label": "./labelsTr/cmda_{:04d}.nii.gz".format(no_f),
    })
    if not os.path.exists(os.path.join(base, "labelsTr/cmda_{:04d}.nii.gz".format(no_f))):
        shutil.copyfile(
            os.path.join(root_t2, os.path.basename(f)),
            os.path.join(base, "imagesTr/cmda_{:04d}_0000.nii.gz".format(no_f)))
        shutil.copyfile(
            os.path.join(root_t1, os.path.basename(f).replace('ceT1', 'Label')),
            os.path.join(base, "labelsTr/cmda_{:04d}.nii.gz".format(no_f)))
    
    no_f += 1000
    data_dict["training"].append({
        "image": "./imagesTr/cmda_{:04d}.nii.gz".format(no_f),
        "label": "./labelsTr/cmda_{:04d}.nii.gz".format(no_f),
    })
    if not os.path.exists(os.path.join(base, "labelsTr/cmda_{:04d}.nii.gz".format(no_f))):
        img = sitk.ReadImage(f)
        arr = np.clip(sitk.GetArrayFromImage(img), 0, 2000)/2000
        img_new = sitk.GetImageFromArray(arr)
        img_new.CopyInformation(img)
        sitk.WriteImage(img_new, os.path.join(base, "imagesTr/cmda_{:04d}_0000.nii.gz".format(no_f)))
        
        shutil.copyfile(
            os.path.join(root_t1, os.path.basename(f).replace('ceT1', 'Label')),
            os.path.join(base, "labelsTr/cmda_{:04d}.nii.gz".format(no_f)))
    

for f in test_list:
    no_f = np.int32(os.path.basename(f).split('_')[-2])
    if '2022' in os.path.basename(f):
        no_f += 500
    
    data_dict["test"].append(
        "./imagesTs/cmda_{:04d}.nii.gz".format(no_f)
    )
    img = sitk.ReadImage(f)
    arr = np.clip(sitk.GetArrayFromImage(img), 0, 1500)/1500
    img_new = sitk.GetImageFromArray(arr)
    img_new.CopyInformation(img)
    sitk.WriteImage(img_new, os.path.join(base, "imagesTs/cmda_{:04d}_0000.nii.gz".format(no_f)))

with open(os.path.join(base, 'dataset.json'), 'w+') as f:
    f.write(json.dumps(data_dict))


for f in allt2_list:
    no_f = np.int32(os.path.basename(f).split('_')[-2])
    if '2022' in os.path.basename(f):
        no_f += 500
    
    img = sitk.ReadImage(f)
    arr = np.clip(sitk.GetArrayFromImage(img), 0, 1500)/1500
    img_new = sitk.GetImageFromArray(arr)
    img_new.CopyInformation(img)
    sitk.WriteImage(img_new, os.path.join(base, "imagesAllT2/cmda_{:04d}_0000.nii.gz".format(no_f)))