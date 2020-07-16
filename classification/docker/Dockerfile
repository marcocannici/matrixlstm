# Download base image ubuntu 16.04
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true
ENV NUM_CORES 10

# Update Ubuntu Software repository
RUN apt-get update
RUN apt-get -y upgrade
RUN apt install -y --no-install-recommends apt-utils

# Build tools.
RUN apt-get update && apt-get install -y build-essential cmake git

# Conda supports delegating to pip to install dependencies
# that aren’t available in anaconda or need to be compiled
# for other reasons. In our case, we need psycopg compiled
# with SSL support. These commands install prereqs necessary
# to build psycopg.
RUN apt-get install -y libpq-dev

# Fonts.
RUN apt-get install -y fonts-liberation

# Utilities.
RUN apt-get install -y \
    bc \
	wget \
    curl \
    libfreetype6-dev \
    libpng12-dev \
    libzmq3-dev \
    pkg-config \
    software-properties-common \
    unzip \
	libsm6 \
	libxext6 \
	libfontconfig1 \
	libxrender1 \
	libav-tools \
        && \
    apt-get clean && \
    apt-get autoremove


RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

ENV PATH "/opt/conda/bin:$PATH"
ENV PATH "/opt/conda/envs:$PATH"
ENV PATH "/opt/conda/envs/py36/bin:$PATH"

# https://github.com/ContinuumIO/docker-images/issues/89
#  update conda and setup environment
RUN conda create -y -n py36 python=3.6 pip && \
    echo ". activate py36" >> ~/.bashrc
#RUN conda update -n base -c defaults conda


# Install extras
COPY requirements.yml /requirements.yml
RUN /bin/bash -c ". activate py36 && conda env update -f=/requirements.yml"
RUN /bin/bash -c ". activate py36 && conda install llvmlite=0.30.0=py36hd408876_0"
RUN /bin/bash -c ". activate py36 && conda install numba=0.46.0=py36h962f231_0"

RUN mkdir /.config
RUN chmod 777 /.config

RUN mkdir /.local
RUN chmod 777 /.local

RUN mkdir /.cache
RUN chmod 777 /.cache

RUN mkdir -p /tmp/torch_extensions/lltm
RUN chmod -R 777 /tmp

CMD ["bash"]
ENV HOME /exp
WORKDIR /exp

