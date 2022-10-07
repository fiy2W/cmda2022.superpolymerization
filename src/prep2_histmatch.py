import os
import SimpleITK as sitk
from glob import glob
import yaml
from preprocess import histMatch


with open('config/data.yaml', 'r') as f:
    config = yaml.safe_load(f)

src_t1 = config['preprocess']['resample']['source']
src_t2 = config['preprocess']['resample']['target']
output_t1 = config['preprocess']['histmatch']['source']
output_t2 = config['preprocess']['histmatch']['target']

if not os.path.exists(output_t1):
    os.makedirs(output_t1)
if not os.path.exists(output_t2):
    os.makedirs(output_t2)

flist1 = glob(os.path.join(src_t1, '*ceT1.nii.gz'))
flist2 = glob(os.path.join(src_t2, '*.nii.gz'))

src1 = sitk.ReadImage(os.path.join(src_t1, config['preprocess']['histmatch']['hm_source']))
src2 = sitk.ReadImage(os.path.join(src_t2, config['preprocess']['histmatch']['hm_target']))

for f in flist1:
    print(f)
    img = sitk.ReadImage(f)
    out = histMatch(img, src1)
    sitk.WriteImage(out, os.path.join(output_t1, os.path.basename(f)))

for f in flist2:
    print(f)
    img = sitk.ReadImage(f)
    out = histMatch(img, src2)
    sitk.WriteImage(out, os.path.join(output_t2, os.path.basename(f)))