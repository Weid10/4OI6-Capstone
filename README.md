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

### Issues

- https://github.com/raspberrypi/picamera2/issues/972
  - /boot/firmware/config.txt: dtoverlay=vc4-kms-v3d -> dtoverlay=vc4-kms-v3d,cma-512
- Camera saves photo 90*
  - 
