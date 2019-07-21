# Remove all stopped Docker containers:   sudo docker rm $(sudo docker ps -a -q)
# Remove all untagged images:             sudo docker rmi $(sudo docker images -q --filter "dangling=true")

# My NVIDIA driver supports only cuda <= 9.0
# FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda:9.0-base-ubuntu16.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    sudo \
    software-properties-common \
    vim \
    emacs \
    wget

# Sometimes needed to avoid SSL CA issues.
RUN update-ca-certificates

ENV HOME /home
WORKDIR ${HOME}/

# Download Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p ${HOME}/miniconda3 && \
    rm miniconda.sh

# Setting these env var outside of the install script to ensure
# they persist in image
# (See https://stackoverflow.com/questions/33379393/docker-env-vs-run-export)
ENV PATH ${HOME}/miniconda3/bin:$PATH
ENV CONDA_PATH ${HOME}/miniconda3
ENV LD_LIBRARY_PATH ${CONDA_PATH}/lib:${LD_LIBRARY_PATH}

# Set JAVA_HOME for Spark
ENV JAVA_HOME ${HOME}/miniconda3

# Install Spark for Horizon
RUN wget https://archive.apache.org/dist/spark/spark-2.3.3/spark-2.3.3-bin-hadoop2.7.tgz && \
    tar -xzf spark-2.3.3-bin-hadoop2.7.tgz && \
    mv spark-2.3.3-bin-hadoop2.7 /usr/local/spark && \
    rm spark-2.3.3-bin-hadoop2.7.tgz

# Add files to image
ADD ./config drl-frameworks/config
ADD ./scripts drl-frameworks/scripts
ADD ./lib drl-frameworks/lib
ADD ./scripts/docker/on_start.sh /usr/local/bin/on_start.sh
RUN chmod 777 /usr/local/bin/on_start.sh

# Set up project environment
WORKDIR ${HOME}/drl-frameworks
RUN bash ./scripts/setup_env.sh

# Define default command.
CMD /usr/local/bin/on_start.sh