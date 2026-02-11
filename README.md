# Fill-osopher


### Getting Started

```
python3 -m venv venv
source ./venv/bin/activate
pip install opencv-contrib-python
# or
sudo apt install -y python3-opencv
sudo apt install -y opencv-data

import cv2
import math
import time
from ultralytics import YOLO
```

https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf

take photo - picamera2
process photo
math to determine volume
output i2c or something

cv2-> imports pyopencv that can link into existing modules like yolo

math-> basic arithmetic operations

time-> grabs timestamp for recording data (might be useful later to calculate delay in response time to stop liquid flow)

yolo-> YouOnlyLookOnce prebuilt object recognition and classification beast

sudo apt update
sudo apt install -y \
  build-essential \
  libssl-dev \
  zlib1g-dev \
  libncurses5-dev \
  libncursesw5-dev \
  libreadline-dev \
  libsqlite3-dev \
  libgdbm-dev \
  libdb5.3-dev \
  libbz2-dev \
  libexpat1-dev \
  liblzma-dev \
  tk-dev \
  libffi-dev \
  wget

cd /usr/src
sudo wget https://www.python.org/ftp/python/3.9.18/Python-3.9.18.tgz
sudo tar xzf Python-3.9.18.tgz
cd Python-3.9.18
sudo ./configure --enable-optimizations
sudo make -j4   # or use -j2 if your Pi is slow
sudo make altinstall

python3.9 -m venv yolopi
source yolopi/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python

wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.onnx



### Run Model on Desktop

```
cd ./4OI6-Capstone
source ./venv/bin/activate
python3 ./danielspi/main.py
```


### Issues

- https://github.com/raspberrypi/picamera2/issues/972
  - /boot/firmware/config.txt: dtoverlay=vc4-kms-v3d -> dtoverlay=vc4-kms-v3d,cma-512
- Camera saves photo 90*
  - rotate image array using numpy
