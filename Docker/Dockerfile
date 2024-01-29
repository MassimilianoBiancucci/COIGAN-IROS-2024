FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

RUN apt-get update 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.8 python3-pip
# Update symlink to point to latest
RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.8 /usr/bin/python3
RUN python3 --version
RUN pip3 --version

#Sistema e basilari
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5dbus5 \
    qttools5-dev \
    qttools5-dev-tools \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libxfixes-dev \
    libx11-xcb-dev \
    libxcb-glx0-dev \
	git

RUN python3 --version
RUN pip3 --version
RUN pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install	torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116


#definisco l'enviroment
ENV APPDIR coigan
WORKDIR coigan

RUN cd /coigan/
RUN git clone https://github.com/MassimilianoBiancucci/COIGAN-controllable-object-inpainting.git
RUN pip install -e /coigan/COIGAN-controllable-object-inpainting/COIGAN/

RUN pip install jupyterlab

RUN mkdir /notebooks
CMD jupyter notebook --no-browser --ip 0.0.0.0 --port 8888  --allow-root /notebooks

