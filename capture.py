
from picamera2 import Picamera2
import time
import numpy as np

WIDTH = 1980
HEIGHT = 1080

def camera_init():
    # Initialize Picamera2 for capturing images
    picam2 = Picamera2()

    camera_config = picam2.create_still_configuration()
    camera_config['main']['format'] = 'BGR888'
    camera_config['main']['size'] = (WIDTH, HEIGHT)
    picam2.configure(camera_config)
    picam2.start()

    time.sleep(2)   # Allow camera to warm up

    return picam2

def take_photo(picam2):
    # Use Picamera2 to take a photo and save it to OUTPUT_PATH/OUTPUT_NAME
    # https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf

    array = picam2.capture_array()
    array = np.rot90(array, k=3)  # Rotate to correct orientation

    return array


def save_photo(array, save_location="samples/test.jpg"):
    # Save a numpy array as an image file
    
    import cv2
    cv2.imwrite(save_location, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    print(f"Photo saved to {save_location}")


if __name__ == "__main__":
    picam2 = camera_init()
    array = take_photo(picam2)
    save_photo(array, save_location="../samples/test.jpg")
    picam2.stop()
