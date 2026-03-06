import cv2
import capture
import yolo
import serial_control
import time

USE_TEST_PHOTO = True
SAVE_PHOTO = True
HEADLESS = True

if __name__ == "__main__":
    
    cam = capture.Camera()
    rgb_array = None
    m = yolo.model()
    serial_control.init_serial()


    while True:

        # Get photo from camera or disk
        if not USE_TEST_PHOTO:
            # Wait for trigger from serial
            serial_control.wait_for_trigger()
            rgb_array = cam.take_photo()
            if SAVE_PHOTO:
                cam.save_photo(rgb_array, save_location="./samples/test.jpg")
        else: 
            print("Using test photo from disk: ./samples/test.jpg")
            rgb_array = cv2.imread("./samples/test.jpg")


        # Analze the photo with YOLO and get the estimated volume
        print("Analyzing photo...")
        disp = m.analyze_frame(rgb_array)

        # set display to the marked image
        disp = m.draw_frame(disp)
        print(f"Estimated volume: {m.vol_final:.1f} ml")

        if SAVE_PHOTO:
            cam.save_photo(disp, save_location="./samples/marked.jpg")

        if not HEADLESS:
            cv2.putText(disp,
                        "Press 'c' for new capture, 'q' to quit",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("CupPiYOLO", disp)


        serial_control.send_volume(m.vol_final)
        time.sleep(10)