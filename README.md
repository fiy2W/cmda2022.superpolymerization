# cmda2022.superpolymerization
Source code for [CrossMoDA 2022 challenge](https://crossmoda2022.grand-challenge.org/), coming soon.

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
    python nnunet/inference/predict_simple.py -i $nnUNet_raw_data_base/Task701_CAMDA1/imagesTs -o $output -t 701 -m 3d_fullres -chk model_best --num_threads_preprocessing 2
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
TBD
```