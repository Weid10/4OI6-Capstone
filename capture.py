
from picamera2 import Picamera2
import time
import numpy as np

WIDTH = 1920
HEIGHT = 1080
DEFAULT_SAVE_PATH = "./samples/test.jpg"

class Camera:

    def __init__(self):
        # Initialize Picamera2 for capturing images
        self.picam2 = Picamera2()

        camera_config = self.picam2.create_still_configuration()
        camera_config['main']['format'] = 'RGB888'
        camera_config['main']['size'] = (WIDTH, HEIGHT)
        self.picam2.configure(camera_config)
        self.picam2.start()

        time.sleep(2)   # Allow camera to warm up


    def take_photo(self):
        # Use Picamera2 to take a photo and save it to OUTPUT_PATH/OUTPUT_NAME
        # https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
    
        rgb = self.picam2.capture_array()
        rgb_corrected = np.rot90(rgb, k=3)  # Rotate to correct orientation

        return rgb_corrected


    def save_photo(self, array, save_location=DEFAULT_SAVE_PATH):
        # Save a numpy array as an image file
        
        import cv2
        cv2.imwrite(save_location, array)
        print(f"Photo saved to {save_location}")


    def __exit__(self):
        self.picam2.stop()


if __name__ == "__main__":
    cam = Camera()
    rgb = cam.take_photo()
    cam.save_photo(rgb, save_location=DEFAULT_SAVE_PATH)
