import cv2
import numpy as np
import math
from ultralytics import YOLO

# load YOLO
model = YOLO("yolov8n.pt")

# calibration
mm_per_pixel = 0.45


def cylinder_volume(h, d):
    r = d/2
    return math.pi*r*r*h/1000


def frustum_volume(h, d1, d2):
    return (math.pi*h*(d1*d1 + d1*d2 + d2*d2))/(12*1000)


def measure_widths(roi):

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 60, 150)

    h, w = edges.shape

    top_band = edges[0:int(h*0.25), :]
    bottom_band = edges[int(h*0.75):h, :]

    top_x = np.where(top_band > 0)
    bot_x = np.where(bottom_band > 0)

    if len(top_x[1]) < 10 or len(bot_x[1]) < 10:
        return None

    top_width = top_x[1].max() - top_x[1].min()
    bot_width = bot_x[1].max() - bot_x[1].min()

    return top_width, bot_width


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    results = model(frame)[0]

    for box in results.boxes:

        cls = int(box.cls[0])
        label = results.names[cls]

        if label != "cup":
            continue

        x1,y1,x2,y2 = map(int,box.xyxy[0])

        cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,0),2)

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        measurement = measure_widths(roi)

        height_px = y2-y1

        if measurement is None:

            width_px = x2-x1

            diameter_mm = width_px * mm_per_pixel
            height_mm = height_px * mm_per_pixel

            volume = cylinder_volume(height_mm, diameter_mm)
            shape = "Cylinder"

        else:

            top_w, bot_w = measurement

            height_mm = height_px * mm_per_pixel
            top_mm = top_w * mm_per_pixel
            bot_mm = bot_w * mm_per_pixel

            ratio = top_w / bot_w

            if 0.9 < ratio < 1.1:

                shape = "Cylinder"

                diameter = (top_mm + bot_mm)/2

                volume = cylinder_volume(height_mm, diameter)

            else:

                shape = "Frustum"

                volume = frustum_volume(height_mm, top_mm, bot_mm)

        cv2.putText(display,
                    shape,
                    (x1,y1-40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,255),
                    2)

        cv2.putText(display,
                    f"Volume {volume:.1f} mL",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,255),
                    2)

    cv2.imshow("Cup Volume", display)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()