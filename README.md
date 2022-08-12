# cmda2022.superpolymerization
Source codes for CrossMoDA 2022 challenge, coming soon.

## Docker

### Inference
We provide Docker Hub image for each task. 
```
Task 1:
blackfeather61/cmda2022.superpolymerization.task1

Task 2:
blackfeather61/cmda2022.superpolymerization.task2
```

Or you can build an inference Docker image by yourself.
```sh
sudo docker build -f Dockerfile -t cmda2022.superpolymerization.inference .
```

Test Docker container. `[input directory]` will be the absolute path of our directory containing the test set, `[output directory]` will be the absolute path of the prediction directory and `[image name]` is the name of Docker image.
```sh
sudo docker run --rm -v [input directory]:/input/:ro -v [output directory]:/output -it [image name]
```
