import cv2
import yolo
import time
import os
import yaml

USE_TEST_PHOTO = True
SAVE_PHOTO = True
HEADLESS = True

if __name__ == "__main__":
    
    rgb_array = None
    m = yolo.model()

    # Load config parameters
    with open("./samples/test/values.yaml", "r") as f:
        values = yaml.safe_load(f)
    # print(values)
    values = values["pictures"] 
    iteration = 0

    while True:

        # Get photo from camera
        file_name = values[iteration]['name']

        print(f"Using test photos from disk: {file_name}")
        rgb_array = cv2.imread(f"./samples/test/{file_name}")


        # Analze the photo with YOLO and get the estimated volume
        print("Analyzing photo...")
        disp = m.analyze_frame(rgb_array)

        # set display to the marked image
        disp = m.draw_frame(disp)
        print(f"Estimated volume: {m.vol_final:.1f} ml")

        if SAVE_PHOTO:
            cv2.imwrite("./samples/marked.jpg", disp)

        if not HEADLESS:
            cv2.putText(disp,
                        "Press 'c' for new capture, 'q' to quit",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("CupPiYOLO", disp)

        # compare values from the image to the expected values from the yaml file
        expected_volume = values[iteration]['volume']
        expected_height = values[iteration]['height']
        expected_diameter = values[iteration]['diameter']
        print(f"Expected volume: {expected_volume:.1f} ml")
        # print(f"Expected dimensions: {expected_height:.1f} mm x {expected_diameter:.1f} mm")


        calculated_volume = m.vol_final
        calculated_height = m.dim_mm[0]
        calculated_diameter = m.dim_mm[1]
        print(f"Calculated volume: {calculated_volume:.1f} ml")
        print(f"Calculated dimensions: {calculated_height:.1f} mm x {calculated_diameter:.1f} mm")

        difference_volume = calculated_volume - expected_volume
        print(f"Volume difference: {difference_volume:.1f} ml, {difference_volume/expected_volume*100:.1f}%")



        iteration = (iteration + 1) % len(file_name)

        # serial_control.send_volume(m.vol_final)
        time.sleep(5)