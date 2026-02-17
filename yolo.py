import cv2
import math
from ultralytics import YOLO
import capture  #  Picamera2 module

# ------------ CONFIG ------------
CONF_THRESH = 0.35

# treat these COCO classes as "containers"
# 41=cup, 70=toilet, 45=bowl, 40=wine glass
CONTAINER_CLASS_IDS = [41, 70, 45, 40]
# --------------------------------

def cylinder_volume_ml(h_mm, d_mm):
    """Approximate cup as cylinder, h,d in mm."""
    h_cm = h_mm / 10.0
    r_cm = (d_mm / 2.0) / 10.0
    return math.pi * r_cm * r_cm * h_cm

print("Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")   # or path to your cust
model.fuse()
names = model.names
print("Loaded.")

cv2.namedWindow("CupPiYOLO")

def nothing(x): pass

# per-axis px/mm (depth vs width)
cv2.createTrackbar("PXmm_H", "CupPiYOLO", 3, 20, nothing)
cv2.createTrackbar("PXmm_W", "CupPiYOLO", 3, 20, nothing)

# global volume scale (x1000) – set once after calibration
cv2.createTrackbar("VolScale_x1000", "CupPiYOLO", 1000, 3000, nothing)


def analyze_frame(frame_bgr):
    """Run YOLO + simple geometry on a single BGR frame."""
    display = frame_bgr.copy()

    pxmm_h = cv2.getTrackbarPos("PXmm_H", "CupPiYOLO")
    pxmm_w = cv2.getTrackbarPos("PXmm_W", "CupPiYOLO")
    vscale = cv2.getTrackbarPos("VolScale_x1000", "CupPiYOLO")

    if pxmm_h < 1: pxmm_h = 1
    if pxmm_w < 1: pxmm_w = 1
    if vscale < 1: vscale = 1

    vol_scale = vscale / 1000.0

    # YOLO inference
    results = model(frame_bgr, conf=CONF_THRESH, verbose=False)

    best_box = None
    best_conf = 0.0
    best_cls  = None

    if len(results):
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in CONTAINER_CLASS_IDS:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            area = (x2 - x1) * (y2 - y1)
            if area <= 0:
                continue

            if conf > best_conf:
                best_conf = conf
                best_box  = (int(x1), int(y1), int(x2), int(y2))
                best_cls  = cls_id

    h_px = w_px = 0
    h_mm = d_mm = None
    vol_raw = vol_final = None

    if best_box is not None:
        x1, y1, x2, y2 = best_box

        label = f"{names[best_cls]} {best_conf:.2f}"
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        raw_w = abs(x2 - x1)
        raw_h = abs(y2 - y1)

        # treat vertical dimension as height
        if raw_h < raw_w:
            raw_h, raw_w = raw_w, raw_h

        h_px = raw_h
        w_px = raw_w

        h_mm = h_px / pxmm_h
        d_mm = w_px / pxmm_w

        vol_raw   = cylinder_volume_ml(h_mm, d_mm)
        vol_final = vol_raw * vol_scale
        print(f"vol_final = {vol_final:.1f} mL")

        cv2.putText(display,
                    f"h_px={h_px}  w_px={w_px}",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(display,
                    f"h_mm={h_mm:.1f}  d_mm={d_mm:.1f}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(display,
                    f"Vol(raw)={vol_raw:.1f} mL  x{vol_scale:.2f}",
                    (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(display,
                    f"Volume ≈ {vol_final:.1f} mL",
                    (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(display,
                    "No container detected",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return display


def main(picam2):
    import numpy as np
    print("Controls: c=capture, q=quit")
    current = None

    while True:
        if current is None:
            # blank screen with instructions until we capture
            blank = 255 * np.ones((480, 640, 3), dtype="uint8")
            cv2.putText(blank,
                        "Press 'c' to capture from Pi Camera, 'q' to quit",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.imshow("CupPiYOLO", blank)
        else:
            disp = analyze_frame(current)
            cv2.putText(disp,
                        "Press 'c' for new capture, 'q' to quit",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("CupPiYOLO", disp)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # take a single photo from Picamera2 (RGB)
            rgb = capture.take_photo(picam2)   # returns numpy array :contentReference[oaicite:3]{index=3}
            if rgb is None:
                continue
            # convert RGB -> BGR for OpenCV/YOLO
            current = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
