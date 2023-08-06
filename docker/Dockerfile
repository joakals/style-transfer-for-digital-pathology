FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update && apt-get install -y \
    python3-opencv ca-certificates python3-dev python3-venv git wget sudo ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && rm get-pip.py

RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install dependencies
RUN pip install tensorboard cmake
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Set a fixed model cache directory
ENV FVCORE_CACHE="/tmp"

## Setup for tiling
# Install missing dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    apt-transport-https

WORKDIR /tmp

RUN add-apt-repository universe
ENV TZ="Europe/Oslo"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN echo "deb [trusted=yes] https://E3NZ_BAkEJdz2HBa1KZL@apt.fury.io/icgipackaging/ /" /etc/apt/sources.list.d/fury.list
RUN apt-get update && apt-get install -yq \
    build-essential \
    git \
    libsm6 \
    libxrender1 \
    libxext6 \
    libglib2.0-0 \
    libgl1 \
    vim

COPY requirements.txt /tmp
RUN pip3 install -r requirements.txt

CMD ["bash"]
