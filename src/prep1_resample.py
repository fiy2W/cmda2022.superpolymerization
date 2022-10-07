import os
import SimpleITK as sitk
from glob import glob
import yaml
from preprocess import resampleDCM


with open('config/data.yaml', 'r') as f:
    config = yaml.safe_load(f)

spacing = config['preprocess']['resample']['spacing']
t1_path = config['data']['train']['source']
t1gif_path = config['data']['train']['GIF']
t2_path1 = config['data']['train']['target']
t2_path2 = config['data']['valid']['target']
output_t1 = config['preprocess']['resample']['source']
output_t2 = config['preprocess']['resample']['target']
output_gif = config['preprocess']['resample']['GIF']

if not os.path.exists(output_t1):
    os.makedirs(output_t1)
if not os.path.exists(output_t2):
    os.makedirs(output_t2)
if not os.path.exists(output_gif):
    os.makedirs(output_gif)

flist = glob(os.path.join(t1_path, '*ceT1.nii.gz')) + \
    glob(os.path.join(t2_path1, '*.nii.gz')) + \
    glob(os.path.join(t2_path2, '*.nii.gz'))

for f in flist:
    print(f)
    img = sitk.ReadImage(f)
    img, _, _ = resampleDCM(img, new_spacing=spacing, is_label=False)
    
    if 'ceT1' in f:
        seg = sitk.ReadImage(f.replace('ceT1', 'Label'))
        seg, _, _ = resampleDCM(seg, new_spacing=spacing, is_label=True)

        gif = sitk.ReadImage(os.path.join(t1gif_path, os.path.basename(f).replace('ceT1', 'GIFoutput')))
        gif, _, _ = resampleDCM(gif, new_spacing=spacing, is_label=True)

        sitk.WriteImage(img, os.path.join(output_t1, os.path.basename(f)))
        sitk.WriteImage(seg, os.path.join(output_t1, os.path.basename(f.replace('ceT1', 'Label'))))
        sitk.WriteImage(gif, os.path.join(output_gif, os.path.basename(f.replace('ceT1', 'gif'))))
    else:
        sitk.WriteImage(img, os.path.join(output_t2, os.path.basename(f)))