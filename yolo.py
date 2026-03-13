import cv2
import math
from ultralytics import YOLO # type: ignore

# ------------ CONFIG ------------
CONF_THRESH = 0.1

# treat these COCO classes as "containers"
# 41=cup, 70=toilet, 45=bowl, 40=wine glass
CONTAINER_CLASS_IDS = [41, 70, 45, 40, 39, 75]

CALIBRATE_DIAMETER = 4.6
CALIBRATE_HEIGHT = 6.5
CALIBRATE_VOLUME  = 950.0
CALIBRATE_DEPTH = 233.0 / 1920.0 # adjust depth scalar (see get_depth_scalar) to improve volume estimates
CALIBRATE_DEPTH_SENSITIVITY = 2.5
# --------------------------------

def cylinder_volume_ml(h_mm, d_mm):
    """
    Approximate cup as cylinder, h,d in mm.
    """
    h_cm = h_mm / 10.0
    r_cm = (d_mm / 2.0) / 10.0
    return math.pi * r_cm * r_cm * h_cm


def frustum_volume_ml(h_mm, d1_mm, d2_mm):
    """
    Approximate cup as frustum, h,d1,d2 in mm.
    """
    h_cm = h_mm / 10.0
    r1_cm = (d1_mm / 2.0) / 10.0
    r2_cm = (d2_mm / 2.0) / 10.0
    return (math.pi * h_cm * (r1_cm*r1_cm + r1_cm*r2_cm + r2_cm*r2_cm)) / 3.0


def get_depth_scalar(best_box):
    """
    Simple approximation of the depth of the cup based on the position of the bounding box from the bottom of the image
    """
    x1, y1, x2, y2 = best_box

    dist = 1920 - y2
    # potentially add multiplier or constant
    depth_scalar = 1.0 + (dist / 1920.0 - CALIBRATE_DEPTH) * CALIBRATE_DEPTH_SENSITIVITY
    return depth_scalar


def get_top_bottom_widths(roi):
    """
    Gets the top and bottom diameter of the cup by edge detection
    """

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # split top and bottom for separate edge detection
    h, w = gray.shape
    top_edge = gray[0:int(h*0.25), :]
    bottom_edge = gray[int(h*0.85):h, :]

    # edge detection
    top_band = cv2.Canny(top_edge, 30, 100)
    bottom_band = cv2.Canny(bottom_edge, 30, 30)
    cv2.imwrite("./samples/top_band.jpg", top_band)
    cv2.imwrite("./samples/bottom_band.jpg", bottom_band)

    # get x coordinates of edges in top and bottom bands
    top_x = np.where(top_band > 0)
    bot_x = np.where(bottom_band > 0)

    if len(top_x[1]) < 10 or len(bot_x[1]) < 10:
        return None

    top_q_low = np.percentile(top_x[1], 5)
    top_q_high = np.percentile(top_x[1], 95)
    bot_q_low = np.percentile(bot_x[1], 5)
    bot_q_high = np.percentile(bot_x[1], 95)

    top_width = top_q_high - top_q_low
    bot_width = bot_q_high - bot_q_low

    print(f"top_width={top_width}  bot_width={bot_width}")

    return top_width, bot_width, (bot_q_low, bot_q_high)


class model:
    def __init__(self):

        self.shape = "Cylinder"

        self.best_box = (0,0,0,0)
        self.best_bound = [(0,0), (0,0), (0,0), (0,0)] # top left, top right, bot left, bot right
        self.best_conf = 0.0
        self.best_cls  = 0

        self.pxmm_h = CALIBRATE_HEIGHT
        self.pxmm_w = CALIBRATE_DIAMETER
        self.vscale = CALIBRATE_VOLUME

        self.dim_px = (0,0)
        self.dim_mm = (0,0)
        
        self.vol_final = 0.0

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
        """
        Run YOLO + simple geometry on a single RGB frame
        """
        display = rgb_frame.copy()

        vol_scale = self.vscale / 1000.0

        # YOLO inference
        results = self.model(rgb_frame, conf=CONF_THRESH, verbose=False)
        if len(results):
            boxes = results[0].boxes
            
            conf_tmp = 0.0
            box_conf = boxes[0]
            for box in boxes:
                cls_id = int(box.cls[0])
                # print(cls_id)
                if cls_id not in CONTAINER_CLASS_IDS:
                    continue

                conf = float(box.conf[0])
                if conf > conf_tmp:
                    conf_tmp = conf
                    box_conf = box

            x1, y1, x2, y2 = box_conf.xyxy[0].cpu().numpy()

            self.best_conf = conf
            self.best_box  = (int(x1), int(y1), int(x2), int(y2))
            self.best_cls  = cls_id
            self.best_bound = [
                (int(x1), int(y1)), # Top-Left
                (int(x2), int(y1)), # Top-Right
                (int(x2), int(y2)), # Bottom-Right
                (int(x1), int(y2))  # Bottom-Left
            ]
        else:
            print("Error: No container detected")
            return False, display


        if self.best_box is not None:
            x1, y1, x2, y2 = self.best_box
            ratio = 0

            depth_scale = get_depth_scalar(self.best_box)

            # get height dimensions, width is handled later
            # scale dimensions by manually calibrated pixel per mm, and depth
            height_px = abs(y2 - y1)
            height_mm = height_px / self.pxmm_h * depth_scale

            # see if top/bottom widths are different
            widths = get_top_bottom_widths(rgb_frame[y1:y2, x1:x2])
            if widths is None:
                # fallback to treating as cylinder if we can't get widths
                self.shape = "Cylinder"
                diameter_px = abs(x2 - x1)
                diameter_mm = diameter_px / self.pxmm_w * depth_scale
                volume = cylinder_volume_ml(height_mm, diameter_mm)
                print(f"Raw volume estimate (cylinder error): {volume:.1f} mL (h={height_mm:.1f}mm, d={diameter_mm:.1f}mm)")

            else:
                top_px, bot_px, bound_adjust = widths
                ratio = top_px / bot_px

                # scale dimensions by manually calibrated pixel per mm, and depth
                top_mm = top_px / self.pxmm_w * depth_scale
                bot_mm = bot_px / self.pxmm_w * depth_scale

                # calculate volume for cylinder or frustum
                if 0.9 < ratio < 1.1:
                    self.shape = "Cylinder"
                    diameter_mm = (top_mm + bot_mm)/2
                    volume = cylinder_volume_ml(height_mm, diameter_mm)
                    print(f"Raw volume estimate (cylinder): {volume:.1f} mL (h={height_mm:.1f}mm, d={diameter_mm:.1f}mm, ratio={ratio:.2f})")

                else:
                    self.shape = "Frustum"
                    volume = frustum_volume_ml(height_mm, top_mm, bot_mm)
                    print(f"Raw volume estimate (frustrum): {volume:.1f} mL (h={height_mm:.1f}mm, top={top_mm:.1f}mm, bot={bot_mm:.1f}mm, ratio={ratio:.2f})")

                    # adjust bounding box to match measured bottom widths
                    self.best_bound[2] = (int(x1 + bound_adjust[1]), y2) # Bottom-Right
                    self.best_bound[3] = (int(x1 + bound_adjust[0]), y2) # Bottom-Left

            # self.dim_px = (height_px, diameter_px)
            # self.dim_mm = (height_mm, diameter_mm)

            self.vol_final = volume * vol_scale
            print(f"vol_final = {self.vol_final:.1f} mL")

        return True, display


    def get_trackbar_values(self):
        self.pxmm_h = cv2.getTrackbarPos("PXmm_H", "CupPiYOLO")
        self.pxmm_w = cv2.getTrackbarPos("PXmm_W", "CupPiYOLO")
        self.vscale = cv2.getTrackbarPos("VolScale_x1000", "CupPiYOLO")
        if self.pxmm_h < 1: self.pxmm_h = 1
        if self.pxmm_w < 1: self.pxmm_w = 1
        if self.vscale < 1: self.vscale = 1


    def draw_info(self, display):
        """
        Draw bounding box and volume estimate on display frame
        """
        x1, y1, x2, y2 = self.best_box
        height_px, diameter_px = self.dim_px
        h_mm, d_mm = self.dim_mm

        label = f"{self.names[self.best_cls]} {self.best_conf:.2f}"
        if self.shape == "Frustum":
            bound = self.best_bound
            cv2.line(display, bound[0], bound[1], (0, 255, 0), 2)
            cv2.line(display, bound[1], bound[2], (0, 255, 0), 2)
            cv2.line(display, bound[2], bound[3], (0, 255, 0), 2)
            cv2.line(display, bound[3], bound[0], (0, 255, 0), 2)
        else:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, label, (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.best_box is not None:
            cv2.putText(display,
                        f"height_px={height_px}  diameter_px={diameter_px}",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(display,
                        f"h_mm={h_mm:.1f}  d_mm={d_mm:.1f}",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(display,
                        f"Volume ≈ {self.vol_final:.1f} mL",
                        (10, 105),
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
    # import capture  #  Picamera2 module
    import numpy as np
    import time

    rgb_frame = None
    # cam = capture.Camera()
    m = model()
    m.init_display()

    # rgb_frame = cv2.imread("samples/test/cup_red0.jpg")
    rgb_frame = cv2.imread("samples/test/cup_red_forward0.jpg")

    while True:
        # rgb_frame = cam.take_photo()

        ret, disp = m.analyze_frame(rgb_frame)
        if not ret:
            print("No container detected, skipping serial send")

        disp = m.draw_info(disp)
        cv2.imwrite("./samples/marked.jpg", disp)
    
        time.sleep(1)
