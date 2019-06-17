FROM nvidia/cuda:10.0-devel-ubuntu16.04
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update
RUN apt-get install -y libgtk2.0-dev

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 2.7 environment
RUN /miniconda/bin/conda install conda-build \
 && /miniconda/bin/conda create -y --name py27 python=2.7 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py27
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Install Dependencies
RUN conda install pytorch=1.0.0 cudatoolkit=10.0 -c pytorch \
 && conda clean -ya

# Install TorchVision
RUN pip install torchvision==0.2.2

ADD requirements.txt /app/
RUN pip install -r requirements.txt
RUN pip install pycocotools

# Build Faster RCNN
ADD . /app
RUN cd lib && rm -rf build && python setup.py build develop

WORKDIR /app
