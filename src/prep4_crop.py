import os
import SimpleITK as sitk
from glob import glob
import yaml
import numpy as np
from preprocess import ants_affine


with open('config/data.yaml', 'r') as f:
    config = yaml.safe_load(f)


src_t1 = config['preprocess']['affine']['source']
src_t2 = config['preprocess']['affine']['target']
src_gif = config['preprocess']['affine']['GIF']
src_valid = config['preprocess']['histmatch']['target']
output_t1 = config['preprocess']['crop']['source']
output_t2 = config['preprocess']['crop']['target']
output_gif = config['preprocess']['crop']['GIF']
output_t2_valid = config['preprocess']['crop']['valid']

if not os.path.exists(output_t1):
    os.makedirs(output_t1)
if not os.path.exists(output_t2):
    os.makedirs(output_t2)
if not os.path.exists(output_gif):
    os.makedirs(output_gif)
if not os.path.exists(output_t2_valid):
    os.makedirs(output_t2_valid)

testlist = [os.path.join(src_valid, os.path.basename(f)) for f in glob(os.path.join(config['data']['valid']['target'], '*.nii.gz'))]
flist = glob(os.path.join(src_t1, '*ceT1.nii.gz')) + \
    glob(os.path.join(src_t2, '*.nii.gz')) + \
    testlist

for f in flist:
    print(f)
    
    img = sitk.ReadImage(f)
    arr = sitk.GetArrayFromImage(img)

    if arr.shape[0] <= 80:
        sd, ed = 0, 80
    else:
        sd = np.clip(np.int64(arr.shape[0]*0.375)-40, 0, None)
        ed = sd + 80
    
    img_crop = sitk.GetImageFromArray(arr[sd:ed, 200:456, 128:384])
    img_crop.SetSpacing(img.GetSpacing())
    
    if 'ceT1' in f:
        seg = sitk.ReadImage(f.replace('ceT1', 'Label'))
        arr = sitk.GetArrayFromImage(seg)
        seg_crop = sitk.GetImageFromArray(arr[sd:ed, 200:456, 128:384])
        seg_crop.SetSpacing(seg.GetSpacing())

        gif = sitk.ReadImage(os.path.join(src_gif, os.path.basename(f).replace('ceT1', 'gif')))
        arr = sitk.GetArrayFromImage(gif)
        gif_crop = sitk.GetImageFromArray(arr[sd:ed, 200:456, 128:384])
        gif_crop.SetSpacing(gif.GetSpacing())

        sitk.WriteImage(img_crop, os.path.join(output_t1, os.path.basename(f)))
        sitk.WriteImage(seg_crop, os.path.join(output_t1, os.path.basename(f).replace('ceT1', 'Label')))
        sitk.WriteImage(gif_crop, os.path.join(output_gif, os.path.basename(f).replace('ceT1', 'gif')))
    elif f in testlist:
        sitk.WriteImage(img_crop, os.path.join(output_t2_valid, os.path.basename(f)))
    else:
        sitk.WriteImage(img_crop, os.path.join(output_t2, os.path.basename(f)))