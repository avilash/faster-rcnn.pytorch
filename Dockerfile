FROM nvidia/cuda:10.0-devel
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /faster_rcnn

RUN apt-get update
RUN apt-get install python python-pip python-tk -y
RUN apt-get install libsm6 libxrender1 libfontconfig1 libxext6 -y

RUN pip install https://download.pytorch.org/whl/cu100/torch-1.0.0-cp27-cp27mu-linux_x86_64.whl
RUN pip install torchvision

ADD requirements.txt /faster_rcnn/
RUN pip install -r requirements.txt

ADD lib /faster_rcnn/lib/
WORKDIR /faster_rcnn/lib
RUN rm -rf build
RUN python setup.py build develop

WORKDIR /faster_rcnn
RUN pip install pycocotools