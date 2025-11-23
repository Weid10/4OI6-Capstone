# Fill-osopher


### Getting Started

```
python3 -m venv venv
source ./venv/bin/activate
pip install opencv-contrib-python
# or
sudo apt install -y python3-opencv
sudo apt install -y opencv-data

```

https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf

take photo - picamera2
process photo
math to determine volume
output i2c or something


### Issues

- https://github.com/raspberrypi/picamera2/issues/972
  - /boot/firmware/config.txt: dtoverlay=vc4-kms-v3d -> dtoverlay=vc4-kms-v3d,cma-512
- Camera saves photo 90*
  - 
