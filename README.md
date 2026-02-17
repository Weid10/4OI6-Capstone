# Fill-osopher

## Files
| File Name   | Library     | Function    | 
|-            |-            |-            |
| main.py     |             | Main function |
| capture.py  | picamera2   | Take photo  | 
| yolo.py     | yolo        | Process photo to determine volume |
| serial.py   | pyserial    | Recieve control signal through serial | 


### Run Model on Desktop

```
cd ./4OI6-Capstone
source ./venv/bin/activate
python3 ./main.py
```


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


### Issues

- https://github.com/raspberrypi/picamera2/issues/972
  - /boot/firmware/config.txt: dtoverlay=vc4-kms-v3d -> dtoverlay=vc4-kms-v3d,cma-512
- Camera saves photo 90*
  - rotate image array using numpy
