# cmda2022.superpolymerization
Source code for [CrossMoDA 2022 challenge](https://crossmoda2022.grand-challenge.org/), coming soon.

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