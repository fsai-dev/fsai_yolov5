FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

ENV MKL_THREADING_LAYER GNU
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    curl \
    ca-certificates curl ffmpeg libsm6 libxext6 \
    git wget ninja-build protobuf-compiler libprotobuf-dev build-essential python3-opencv tmux nvidia-container-toolkit cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Set Python 3.10 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
# Install pip for Python 3.10
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.10 get-pip.py && rm get-pip.py
RUN ln -sv /usr/bin/python3 /usr/bin/python


# Install the required python packages globally
ENV PATH="$PATH:/root/.local/bin"

RUN \
    # Upgrade pip
    pip install --upgrade pip && \
    mkdir -p /code

# Set the current working directory
WORKDIR /code

ENV FORCE_CUDA="1"

RUN pip install --user torch torchvision setuptools pycocotools comet_ml

# RUN pip install --user termcolor==1.1.0 numpy==1.23.0 idna==3.7

# Copy the application into the container
COPY ./ ./

RUN pip install --user -r requirements.txt