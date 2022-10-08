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
### Train MSF-Net
```sh
python src/train_msf_cross25dseg_gif.py -d cuda -b 1 -e 1000 -l 2e-4 -s ckpt/msf -v vis/msf
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