import cv2
import math
from ultralytics import YOLO # type: ignore
import numpy as np

# ------------ CONFIG ------------
CONF_THRESH = 0.1

# treat these COCO classes as "containers"
# 41=cup, 70=toilet, 45=bowl, 40=wine glass
CONTAINER_CLASS_IDS = [41, 70, 45, 40, 39, 75]

CALIBRATE_DIAMETER = 5.2
CALIBRATE_HEIGHT = 6.5
CALIBRATE_VOLUME  = 0.85
CALIBRATE_DEPTH = 210.0 / 1920.0 # adjust depth scalar (see get_depth_scalar) to improve volume estimates
CALIBRATE_DEPTH_SENSITIVITY = 2.2

DEBUG = True
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


# -----------------------------
# MASK AND HANDLE REMOVAL LOGIC
# -----------------------------
def get_mask(roi):
    """
    Extracts a clean binary mask of the object in the ROI.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    if DEBUG: cv2.imwrite("samples/debug/blur.jpg", blur)

    edges = cv2.Canny(blur, 40, 120)
    if DEBUG: cv2.imwrite("samples/debug/edges.jpg", edges)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mask = cv2.bitwise_or(edges, th)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    if DEBUG: cv2.imwrite("samples/debug/mask.jpg", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        clean = np.zeros_like(mask)
        cv2.drawContours(clean, [largest], -1, 255, -1)
        if DEBUG: cv2.imwrite("samples/debug/output.jpg", clean)
        return clean

    return mask


def get_width(mask, y):
    """
    Finds the width of the mask at a specific y-row using percentiles to ignore handles/outliers.
    """
    xs = np.where(mask[y] > 0)[0]
    if len(xs) < 20:
        return None

    left = np.percentile(xs, 10)
    right = np.percentile(xs, 90)

    return int(right - left)


def get_true_center(mask):
    """
    Calculates the true center of the cup by taking the global percentiles of all 'x' pixels.
    """
    xs = np.where(mask > 0)[1]

    if len(xs) < 50:
        return None

    left = np.percentile(xs, 20)
    right = np.percentile(xs, 80)

    return int((left + right) / 2)


def get_simple_widths(roi):
    """
    Gets the top and bottom diameter for cups with no handles
    """

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # split top and bottom for separate edge detection
    h, w = blur.shape
    top_edge = blur[0:int(h*0.25), :]
    bottom_edge = blur[int(h*0.85):h, :]

    # edge detection
    top_band = cv2.Canny(top_edge, 60, 120)
    bottom_band = cv2.Canny(bottom_edge, 60, 50)
    if DEBUG:
        cv2.imwrite("./samples/debug/top_band.jpg", top_band)
        cv2.imwrite("./samples/debug/bottom_band.jpg", bottom_band)

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
        cv2.createTrackbar("VolScale", "CupPiYOLO", 1000, 3000, nothing)


    def get_bounding_box(self, results):
        conf_tmp = 0.0
        box_conf = None
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            
            if cls_id not in CONTAINER_CLASS_IDS:
                continue

            conf = float(box.conf[0])
            if conf > conf_tmp:
                conf_tmp = conf
                box_conf = box

        if box_conf is None:
            return None

        x1, y1, x2, y2 = box_conf.xyxy[0].cpu().numpy()
        self.best_box = (int(x1), int(y1), int(x2), int(y2))
        self.best_bound = [
            (x1, y1), (x2, y1), (x2, y2), (x1, y2)
        ]
        return None
    

    def analyze_frame(self, rgb_frame):
        """
        Run YOLO + mask profiling geometry on a single RGB frame
        """
        display = rgb_frame.copy()

        # YOLO inference
        results = self.model(rgb_frame, conf=CONF_THRESH, verbose=False)
        self.get_bounding_box(results)
        
        if self.best_box is None:
            print("Error: No container detected")
            return False, display

        x1, y1, x2, y2 = self.best_box
        roi = rgb_frame[y1:y2, x1:x2]

        ratio = (y2 - y1) / (x2 - x1)
        print(f"Detected box ratio (h/w): {ratio:.2f}")

        if ratio < 1.2:
            print("Applying handle volume calculation")
            self.calculate_volume_handle(roi)
        else:
            print("Applying no-handle volume calculation")
            self.calculate_volume_nohandle(roi)

        return True, display


    def calculate_volume_handle(self, roi):
        """
        Run profiling geometry on a single RGB frame
        """
        if self.best_box is None:
            print("Error: No container detected")
            return False, roi

        x1, y1, x2, y2 = self.best_box
        depth_scale = get_depth_scalar(self.best_box)
        
        mask = get_mask(roi)
        
        height_px = abs(y2 - y1)
        h, w = mask.shape

        widths = []
        for y in range(int(h * 0.1), int(h * 0.9)):
            w_ = get_width(mask, y)
            if w_ is not None:
                widths.append(w_)

        # Fallback if masking fails to find enough rows
        if len(widths) < 20:
            self.shape = "CYL"
            diameter_px = abs(x2 - x1)
            diameter_mm = diameter_px / self.pxmm_w * depth_scale
            height_mm = height_px / self.pxmm_h * depth_scale

            volume = cylinder_volume_ml(height_mm, diameter_mm)
            print(f"Raw volume estimate (fallback): {volume:.1f} mL (h={height_mm:.1f}mm, d={diameter_mm:.1f}mm)")

            self.dim_px = (height_px, diameter_px)
            self.dim_mm = (height_mm, diameter_mm)

        else:
            widths_arr = np.array(widths)

            # Get stable top and bottom diameters
            top_px = np.median(widths_arr[:len(widths_arr)//4])
            bot_px = np.median(widths_arr[-len(widths_arr)//4:])
            
            # Get true center
            cx_local = get_true_center(mask)
            if cx_local is None:
                cx_local = int((x2 - x1) / 2) # fallback to bbox center
            
            cx = x1 + cx_local

            # Rim position offset
            y_top = y1 + int(0.02 * height_px)
            y_bottom = y2
            
            adj_height_px = y_bottom - y_top

            # Scale dimensions by manually calibrated pixel per mm and depth
            adj_height_mm = adj_height_px / self.pxmm_h * depth_scale
            top_mm = top_px / self.pxmm_w * depth_scale
            bot_mm = bot_px / self.pxmm_w * depth_scale

            # Calculate shape logic based on ratio difference
            ratio = abs(top_px - bot_px) / max(top_px, bot_px)

            if ratio < 0.08:
                self.shape = "CYL"
                diameter_mm = (top_mm + bot_mm)/2
                volume = cylinder_volume_ml(adj_height_mm, diameter_mm)
                print(f"Raw volume estimate ({self.shape}): {volume:.1f} mL (h={adj_height_mm:.1f}mm, d={diameter_mm:.1f}mm, ratio={ratio:.3f})")
            else:
                self.shape = "FRUSTUM"
                volume = frustum_volume_ml(adj_height_mm, top_mm, bot_mm)
                print(f"Raw volume estimate ({self.shape}): {volume:.1f} mL (h={adj_height_mm:.1f}mm, top={top_mm:.1f}mm, bot={bot_mm:.1f}mm, ratio={ratio:.3f})")

            # Set bound polygon for drawing
            top_half = int(top_px / 2)
            bot_half = int(bot_px / 2)

            self.best_bound = [
                (cx - top_half, y_top),
                (cx + top_half, y_top),
                (cx + bot_half, y_bottom),
                (cx - bot_half, y_bottom)
            ]

            self.dim_px = (adj_height_px, top_px)
            self.dim_mm = (adj_height_mm, top_mm)

        self.vol_final = volume * self.vscale
        print(f"vol_final = {self.vol_final:.1f} mL")


    def calculate_volume_nohandle(self, roi):
        """
        Run profiling geometry on a single RGB frame assuming no handles (treat as cylinder or frustum)
        """
        if self.best_box is None:
            print("Error: No container detected")
            return False, roi
    
        x1, y1, x2, y2 = self.best_box
        ratio = 0

        depth_scale = get_depth_scalar(self.best_box)

        # get height dimensions, width is handled later
        # scale dimensions by manually calibrated pixel per mm, and depth
        height_px = abs(y2 - y1)
        height_mm = height_px / self.pxmm_h * depth_scale

        # see if top/bottom widths are different
        widths = get_simple_widths(roi)
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

        self.vol_final = volume * self.vscale
        print(f"vol_final = {self.vol_final:.1f} mL")


    def get_trackbar_values(self):
        self.pxmm_h = cv2.getTrackbarPos("PXmm_H", "CupPiYOLO")
        self.pxmm_w = cv2.getTrackbarPos("PXmm_W", "CupPiYOLO")
        vscale_x1000 = cv2.getTrackbarPos("VolScale", "CupPiYOLO")
        self.vscale = vscale_x1000 / 1000.0
        if self.pxmm_h < 1: self.pxmm_h = 1
        if self.pxmm_w < 1: self.pxmm_w = 1
        if self.vscale < 1: self.vscale = 1


    def draw_info(self, display):
        """
        Draw bounding box, bounds, and volume estimate on display frame
        """

        if self.best_box is None:
            cv2.putText(display, "No container detected", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return display

        x1, y1, x2, y2 = self.best_box
        label = f"Estimated: {self.vol_final:.1f} mL"

        # Draw translucent polygon overlay for the measured area
        bound_pts = np.array(self.best_bound, np.int32)
        overlay = display.copy()
        cv2.fillPoly(overlay, [bound_pts], (255, 0, 0))
        cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
        cv2.polylines(display, [bound_pts], True, (255, 0, 0), 2)

        # Draw YOLO bounding box lightly for reference
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(display, label, (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(display,
                    f"Volume ≈ {self.vol_final:.1f} mL ({self.shape})",
                    (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return display


    def __exit__(self):
        cv2.destroyAllWindows()
        pass


if __name__ == "__main__":
    # import capture  #  Picamera2 module
    import time

    rgb_frame = None
    # cam = capture.Camera()
    m = model()
    # m.init_display()

    # rgb_frame = cv2.imread("samples/test/cup_red0.jpg")
    # rgb_frame = cv2.imread("samples/cup_cat_forward_center.jpg") # 1.12
    # rgb_frame = cv2.imread("samples/cup_cat_back_center.jpg")
    rgb_frame = cv2.imread("samples/cup_red_back_center.jpg") # 1.3
    # rgb_frame = cv2.imread("samples/cup_red_middle_center.jpg") # 1.32
    # rgb_frame = cv2.imread("samples/cup_red_forward_center.jpg") # 1.33
    # rgb_frame = cv2.imread("samples/cup_green_back_center.jpg") # 1.08
    # rgb_frame = cv2.imread("samples/test.jpg") # 1

    if rgb_frame is not None:
        # rgb_frame = cam.take_photo()

        ret, disp = m.analyze_frame(rgb_frame)
        if not ret:
            print("No container detected, skipping serial send")

        disp = m.draw_info(disp)
        cv2.imwrite("./samples/marked.jpg", disp)
        
        # Show output for testing
        # cv2.imshow("Result", disp)

    else:
        print("Could not load image. Please check the file path.")