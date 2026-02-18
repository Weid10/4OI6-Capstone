import cv2
import math
from ultralytics import YOLO # type: ignore

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


class model:
    def __init__(self):

        self.best_box = None
        self.best_conf = 0.0
        self.best_cls  = None

        self.pxmm_h = 3
        self.pxmm_w = 3
        self.vscale = 1000

        self.dim_px = None
        self.dim_mm = None
        
        self.vol_raw = None
        self.vol_final = None

        print("Loading YOLOv8n model...")
        self.model = YOLO("yolov8n.pt")
        self.model.fuse()
        self.names = self.model.names
        print("Loaded model")


    def init_display(self):
        cv2.namedWindow("CupPiYOLO")
        def nothing(x): pass

        # per-axis px/mm (depth vs width)
        cv2.createTrackbar("PXmm_H", "CupPiYOLO", 3, 20, nothing)
        cv2.createTrackbar("PXmm_W", "CupPiYOLO", 3, 20, nothing)

        # global volume scale (x1000) – set once after calibration
        cv2.createTrackbar("VolScale_x1000", "CupPiYOLO", 1000, 3000, nothing)


    def analyze_frame(self, rgb_frame):
        """Run YOLO + simple geometry on a single RGB frame."""
        display = rgb_frame.copy()

        vol_scale = self.vscale / 1000.0

        # YOLO inference
        results = self.model(rgb_frame, conf=CONF_THRESH, verbose=False)


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

                # run every time for now
                # if conf > self.best_conf:
                self.best_conf = conf
                self.best_box  = (int(x1), int(y1), int(x2), int(y2))
                self.best_cls  = cls_id

        h_px = w_px = 0
        h_mm = d_mm = None

        if self.best_box is not None:
            x1, y1, x2, y2 = self.best_box

            raw_w = abs(x2 - x1)
            raw_h = abs(y2 - y1)

            # treat vertical dimension as height
            if raw_h < raw_w:
                raw_h, raw_w = raw_w, raw_h

            h_px = raw_h
            w_px = raw_w

            h_mm = h_px / self.pxmm_h
            d_mm = w_px / self.pxmm_w

            self.dim_px = (h_px, w_px)
            self.dim_mm = (h_mm, d_mm)

            self.vol_raw   = cylinder_volume_ml(h_mm, d_mm)
            self.vol_final = self.vol_raw * vol_scale
            print(f"vol_final = {self.vol_final:.1f} mL")

        return display


    def get_display(self):
        self.pxmm_h = cv2.getTrackbarPos("PXmm_H", "CupPiYOLO")
        self.pxmm_w = cv2.getTrackbarPos("PXmm_W", "CupPiYOLO")
        self.vscale = cv2.getTrackbarPos("VolScale_x1000", "CupPiYOLO")
        if self.pxmm_h < 1: self.pxmm_h = 1
        if self.pxmm_w < 1: self.pxmm_w = 1
        if self.vscale < 1: self.vscale = 1


    def set_display(self, display):
        x1, y1, x2, y2 = self.best_box
        h_px, w_px = self.dim_px
        h_mm, d_mm = self.dim_mm

        label = f"{self.names[self.best_cls]} {self.best_conf:.2f}"
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, label, (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.best_box is not None:
            cv2.putText(display,
                        f"h_px={h_px}  w_px={w_px}",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(display,
                        f"h_mm={h_mm:.1f}  d_mm={d_mm:.1f}",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(display,
                        f"Vol(raw)={self.vol_raw:.1f} mL  x{self.vscale/1000.0:.2f}",
                        (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(display,
                        f"Volume ≈ {self.vol_final:.1f} mL",
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display,
                    "No container detected",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return display

    def __exit__(self):
        cv2.destroyAllWindows()
        pass


if __name__ == "__main__":
    import capture  #  Picamera2 module
    import numpy as np

    rgb_frame = None
    cam = capture.Camera()
    m = model()
    m.init_display()

    print("Controls: c=capture, q=quit")

    while True:
        
        m.get_display()
        if rgb_frame is None:
            # blank screen with instructions until we capture
            blank = 255 * np.ones((480, 640, 3), dtype="uint8")
            cv2.putText(blank,
                        "Press 'c' to capture from Pi Camera, 'q' to quit",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.imshow("CupPiYOLO", blank)
        else:
            disp = m.analyze_frame(rgb_frame)
            cv2.putText(disp,
                        "Press 'c' for new capture, 'q' to quit",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            disp = m.set_display(disp)
            cv2.imshow("CupPiYOLO", disp)

        key = cv2.waitKey(3000) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # take a single photo from Picamera2 (RGB)
            rgb_frame = cam.take_photo()
            if rgb_frame is None:
                continue
            # convert RGB -> BGR for OpenCV/YOLO
            # current = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)