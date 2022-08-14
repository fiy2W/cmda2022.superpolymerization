## Pull from existing image
FROM nvcr.io/nvidia/pytorch:21.05-py3
FROM continuumio/miniconda
#FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

FROM ubuntu:latest
WORKDIR /app
ADD . /app
RUN set -xe \
    && apt-get update \
    && apt-get install -y python3-pip
#RUN apt-get update && apt-get install -y python3-pip

## Copy requirements
COPY ../../requirements.txt .
COPY ../../nnunet/ ./
COPY ../../setup.py .

## Install Python packages in Docker image
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install -r requirements.txt
RUN pip3 install -e .

## Copy all files
COPY ../../src ./
COPY ../../ckpt/ ./
COPY ../../atlas/ ./

RUN mkdir /myhome/
RUN chmod 777 /myhome
RUN export HOME=/myhome/

ENV RESULTS_FOLDER=./ckpt

## Execute the inference command 
CMD ["./run_inference.py"]
ENTRYPOINT ["python3"]