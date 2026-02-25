import cv2
import capture
import yolo
import serial_control

USE_TEST_PHOTO = True
SAVE_PHOTO = False
HEADLESS = True

if __name__ == "__main__":
    
    cam = capture.Camera()
    rgb_array = None
    cam = capture.Camera()
    m = yolo.model()
    serial_control.init_serial()


    while True:

        serial_control.wait_for_trigger()

        if not USE_TEST_PHOTO:
            rgb_array = cam.take_photo()
            if SAVE_PHOTO:
                cam.save_photo(rgb_array, save_location="./samples/test.jpg")
        else: 
            print("Using test photo from disk: ./samples/test.jpg")
            rgb_array = cv2.imread("./samples/test.jpg")

        if HEADLESS:
            print("Analyzing photo...")
            m.analyze_frame(rgb_array)
            print(f"Estimated volume: {m.vol_final:.1f} ml")
        else:
            disp = m.analyze_frame(rgb_array)
            cv2.putText(disp,
                        "Press 'c' for new capture, 'q' to quit",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            disp = m.set_display(disp)
            cv2.imshow("CupPiYOLO", disp)

        serial_control.send_volume(m.vol_final)