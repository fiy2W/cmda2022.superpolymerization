import os
import SimpleITK as sitk
from glob import glob
import yaml
from preprocess import ants_affine


with open('config/data.yaml', 'r') as f:
    config = yaml.safe_load(f)


path_img_T1 = config['preprocess']['histmatch']['source']
path_img_T2 = config['preprocess']['histmatch']['target']
fix_T2_name = config['preprocess']['affine']['atlas']
save_path_t1 = config['preprocess']['affine']['df_source']
save_path_t2 = config['preprocess']['affine']['df_target']
if not os.path.exists(save_path_t1):
    os.makedirs(save_path_t1)
if not os.path.exists(save_path_t2):
    os.makedirs(save_path_t2)

mov_T1_names = glob(os.path.join(path_img_T1, '*.nii.gz'))
mov_T2_names = glob(os.path.join(path_img_T2, '*.nii.gz'))

for mov_T1_name in mov_T1_names:
    out_file_name = os.path.basename(mov_T1_name)
    out_T1_name = os.path.join(save_path_t1, out_file_name)
    cmd = 'ANTS 3 -m MI[{},{},1,128] -i 1 -s 4x2x1 -o {} --continue-affine true' \
        .format(os.path.join(path_img_T2, fix_T2_name), mov_T1_name, out_T1_name)
    os.system(cmd)

for mov_T2_name in mov_T2_names:
    out_file_name = os.path.basename(mov_T2_name)
    out_T2_name = os.path.join(save_path_t2, out_file_name)
    cmd = 'ANTS 3 -m CC[{},{},1,2] -i 1 -s 4x2x1 -o {} --continue-affine true' \
        .format(os.path.join(path_img_T2, fix_T2_name), mov_T2_name, out_T2_name)
    os.system(cmd)


path_gif_T1 = config['preprocess']['resample']['GIF']
df_path_t1 = config['preprocess']['affine']['df_source']
df_path_t2 = config['preprocess']['affine']['df_target']
save_path_t1 = config['preprocess']['affine']['source']
save_path_t2 = config['preprocess']['affine']['target']
save_path_gif = config['preprocess']['affine']['GIF']
if not os.path.exists(save_path_t1):
    os.makedirs(save_path_t1)
if not os.path.exists(save_path_t2):
    os.makedirs(save_path_t2)
if not os.path.exists(save_path_gif):
    os.makedirs(save_path_gif)


fixed = sitk.ReadImage(os.path.join(path_img_T2, fix_T2_name))
for mov_T1_name in mov_T1_names:
    out_file_name = os.path.basename(mov_T1_name)
    affine_name = os.path.join(df_path_t1, out_file_name.split('.nii.gz')[0]+'Affine.txt')
    out_T1_name = os.path.join(save_path_t1, out_file_name)
    moving = sitk.ReadImage(mov_T1_name)
    seg = sitk.ReadImage(os.path.join(path_img_T1, out_file_name.split('ceT1.nii.gz')[0] + 'Label.nii.gz'))
    gif = sitk.ReadImage(os.path.join(path_gif_T1, out_file_name.split('ceT1.nii.gz')[0] + 'gif.nii.gz'))
    affined_img = ants_affine(affine_name, moving, fixed, is_label=False)
    affined_seg = ants_affine(affine_name, seg, fixed, is_label=True)
    affined_gif = ants_affine(affine_name, gif, fixed, is_label=True)
    sitk.WriteImage(affined_img, os.path.join(save_path_t1, out_file_name))
    sitk.WriteImage(affined_seg, os.path.join(save_path_t1, out_file_name.split('ceT1.nii.gz')[0] + 'Label.nii.gz'))
    sitk.WriteImage(affined_gif, os.path.join(save_path_gif, out_file_name.split('ceT1.nii.gz')[0] + 'gif.nii.gz'))

for mov_T2_name in mov_T2_names:
    out_file_name = os.path.basename(mov_T2_name)
    affine_name = os.path.join(df_path_t2, out_file_name.split('.nii.gz')[0]+'Affine.txt')
    out_T2_name = os.path.join(save_path_t2, out_file_name)
    moving = sitk.ReadImage(mov_T2_name)
    affined_img = ants_affine(affine_name, moving, fixed, is_label=False)
    sitk.WriteImage(affined_img, os.path.join(save_path_t2, out_file_name))