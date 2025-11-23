from picamera2 import Picamera2
import time


WIDTH = 1980
HEIGHT = 1080


def take_photo(save_location="samples/test.jpg"):
    # Use Picamera2 to take a photo and save it to OUTPUT_PATH/OUTPUT_NAME
    # https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf

    picam2 = Picamera2()

    camera_config = picam2.create_still_configuration()
    camera_config['main']['format'] = 'RGB888'
    # camera_config['main']['size'] = (WIDTH, HEIGHT)
    picam2.configure(camera_config)
    picam2.start()

    time.sleep(2)

    array = None
    if save_location is None:
        array = picam2.capture_array()
    else:
        picam2.capture_file(save_location)
        print(f"Photo taken and saved to {save_location}")

    picam2.stop()

    return array
