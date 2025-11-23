from picamera2 import Picamera2
import time

OUTPUT_PATH = "samples"
OUTPUT_NAME = "test.jpg"

def take_photo():
    picam2 = Picamera2()
    camera_config = picam2.create_still_configuration()
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(2)
    picam2.capture_file(f"{OUTPUT_PATH}/{OUTPUT_NAME}")
    picam2.stop()
    print(f"Photo taken and saved to {OUTPUT_PATH}/{OUTPUT_NAME}")