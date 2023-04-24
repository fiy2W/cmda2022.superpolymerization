# cmda2022.superpolymerization
Source code for the implementation of group `SuperPolymerization` in [CrossMoDA 2022 challenge](https://crossmoda2022.grand-challenge.org/).
More details can be found [here](https://arxiv.org/abs/2210.04255).

## Abstract
Domain adaptation has been widely adopted to transfer styles across multi-vendors and multi-centers, as well as to complement the missing modalities. In this challenge, we proposed an unsupervised domain adaptation framework for cross-modality vestibular schwannoma (VS) and cochlea segmentation and Koos grade prediction. We learn the shared representation from both ceT1 and hrT2 images and recover another modality from the latent representation, and we also utilize proxy tasks of VS segmentation and brain parcellation to restrict the consistency of image structures in domain adaptation. After generating missing modalities, the nnU-Net model is utilized for VS and cochlea segmentation, while a semi-supervised contrastive learning pre-train approach is employed to improve the model performance for Koos grade prediction. On CrossMoDA validation phase Leaderboard, our method received rank 4 in task1 with a mean Dice score of 0.8394 and rank 2 in task2 with Macro-Average Mean Square Error of 0.3941.

## Method
![Overview of the proposed unsupervised domain adaptation segmentation and classification framework.](/docs/framework.png)

## Install
- Install [ANTs](https://github.com/ANTsX/ANTs)
- Install PyTorch
    ```sh
    pip install torch torchvision torchaudio
    ```
- Install requirements
    ```sh
    pip install -r requirements.txt
    ```
- Install nnU-Net from source
    ```
    pip install -e .
    ```

## Training process
### Preprocess image
- Modify data root in [data configuration](config/data.yaml).
- Run the following steps for data preprocessing.
    ```sh
    # resample
    python src/prep1_resample.py

    # histogram matching
    python src/prep2_histmatch.py

    # affine
    python src/prep3_affine.py

    # crop
    python src/prep4_crop.py
    ```
### Domain adaptation
- Train MSF-Net
    ```sh
    python src/train_msf_cross25dseg_gif.py -d cuda -b 1 -e 1000 -l 2e-4 -s ckpt/msf -v vis/msf
    ```
- Generate fake ceT1 and fake hrT2
    ```sh
    python src/test_msf25dseg_gif.py -d cuda -c ckpt/msf/ckpt_1000.pth
    ```
### VS segmentation
- Make nnU-Net dataset
    ```sh
    python src/make_nnunet_set.py
    ```
- Train nnU-Net
    ```sh
    for i in 0 1 2 3 4
    do
        python nnunet/run/run_training.py 3d_fullres nnUNetTrainerV2 Task701_CMDA1 $i
    done
    ```
- Inference
    ```sh
    ## inference test data
    python nnunet/inference/predict_simple.py -i nnUNet_raw_data_base/Task701_CMDA1/imagesTs -o preprocess/b2_nnunet_seg/test -t 701 -m 3d_fullres -chk model_best --num_threads_preprocessing 2

    ## inference training data for Koos grade prediction
    python nnunet/inference/predict_simple.py -i nnUNet_raw_data_base/Task701_CMDA1/imagesAllT2 -o preprocess/b2_nnunet_seg/train_t2 -t 701 -m 3d_fullres -chk model_best --num_threads_preprocessing 2
    ```

### Koos grade prediction
- Semi-contrastive learning pretraining
    ```sh
    python src/train_msf_koo25dseg_contrast.py -d cuda -c ckpt/msf/ckpt_1000.pth -e 1000 -l 1e-2 -s ckpt/msf_koos_contrast -v vis/msf_koos_contrast
    ```
- Train
    ```sh
    for i in 0 1 2 3 4
    do
        python src/train_msf_koo25dseg_contrast_fc.py -d cuda -m ckpt/msf/ckpt_1000.pth -c ckpt/msf_koos_contrast/ckpt_100.pth -e 20 -l 1e-4 -s ckpt/msf_koos_contrast_fc/$i -v vis/msf_koos_contrast_fc/$i -f $i
    done
    ```

## Docker

### Inference
We provide Docker Hub image for each task. 
```sh
#Task 1:
docker pull blackfeather61/cmda2022.superpolymerization.task1

#Task 2:
docker pull blackfeather61/cmda2022.superpolymerization.task2
```

Or you can build an inference Docker image by yourself.
```sh
cd docker/inference/
docker build -f Dockerfile -t cmda2022.superpolymerization.inference .
```

Test Docker container. `[input directory]` will be the absolute path of our directory containing the test set, `[output directory]` will be the absolute path of the prediction directory and `[image name]` is the name of Docker image.
```sh
docker run --gpus all --rm -v [input directory]:/input/:ro -v [output directory]:/output -it [image name]
```

## Citation
If this work is helpful for you, please cite our paper as follows:
```bib
@article{han2022unsupervised,
  title={Unsupervised cross-modality domain adaptation for vestibular schwannoma segmentation and koos grade prediction based on semi-supervised contrastive learning},
  author={Han, Luyi and Huang, Yunzhi and Tan, Tao and Mann, Ritse},
  journal={arXiv preprint arXiv:2210.04255},
  year={2022}
}
```