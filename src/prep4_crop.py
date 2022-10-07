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
output_t1 = config['preprocess']['crop']['source']
output_t2 = config['preprocess']['crop']['target']
output_gif = config['preprocess']['crop']['GIF']

if not os.path.exists(output_t1):
    os.makedirs(output_t1)
if not os.path.exists(output_t2):
    os.makedirs(output_t2)
if not os.path.exists(output_gif):
    os.makedirs(output_gif)

flist = glob(os.path.join(src_t1, '*ceT1.nii.gz')) + \
    glob(os.path.join(src_t2, '*.nii.gz'))

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
        sitk.WriteImage(seg_crop, os.path.join(output_gif, os.path.basename(f).replace('ceT1', 'gif')))
    else:
        sitk.WriteImage(img_crop, os.path.join(output_t2, os.path.basename(f)))