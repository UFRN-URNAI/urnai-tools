FROM tensorflow/tensorflow:2.6.1-gpu-jupyter

env DEBIAN_FRONTEND=noninteractive
env SDL_VIDEODRIVER=dummy
env LANG=en_US.UTF-8
env LC_ALL=en_US.UTF-8
env PYTHONIOENCODING=utf8

# Fix encoding problems
RUN apt-get clean &&\
    apt-get update &&\
    apt-get install -y locales
RUN locale-gen "en_US.UTF-8"
RUN dpkg-reconfigure locales

# Needed by vizdoom, deeprts and gym
RUN apt-get install -y wget \
    git \
    python3 \
    python3-pip \
    build-essential \
    zlib1g-dev \
    libsdl2-dev \
    libjpeg-dev \
    nasm \
    tar \
    libbz2-dev \
    libgtk2.0-dev \
    cmake \
    git \
    libfluidsynth-dev \
    libgme-dev \
    libopenal-dev \
    timidity \
    libwildmidi-dev \
    unzip \
    build-essential \
    zlib1g-dev \
    libsdl2-dev \
    libjpeg-dev \
    nasm \
    tar \
    libbz2-dev \
    libgtk2.0-dev \
    cmake \
    git \
    libfluidsynth-dev \
    libgme-dev \
    libopenal-dev \
    timidity \
    libwildmidi-dev \
    unzip \
    libboost-all-dev
RUN apt-get build-dep python-pygame -y

# Install requirements
RUN pip3 install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Install URNAI
RUN pip3 install --no-cache-dir --no-dependencies git+https://github.com/UFRN-URNAI/urnai-tools@master

# Set environment variables
ENV WORKDIR=/app
WORKDIR ${WORKDIR}
