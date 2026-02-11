import cv2
import numpy as np

# Load ONNX YOLOv5 model
net = cv2.dnn.readNetFromONNX("yolov5n.onnx")

# Initialize camera (PiCam or USB cam)
cap = cv2.VideoCapture(0)

CONFIDENCE_THRESHOLD = 0.4

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    h, w = frame.shape[:2]

    # Preprocess image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()[0]  # YOLOv5 returns a single array

    # Post-processing
    boxes = []
    confidences = []

    for detection in outputs:
        conf = detection[4]
        if conf > CONFIDENCE_THRESHOLD:
            class_conf = detection[5:]
            class_id = np.argmax(class_conf)
            if class_id == 39:  # CLASS 39 = cup in COCO dataset
                center_x, center_y, width, height = detection[0:4]
                x = int((center_x - width / 2) * w / 640)
                y = int((center_y - height / 2) * h / 640)
                width = int(width * w / 640)
                height = int(height * h / 640)
                boxes.append([x, y, width, height])
                confidences.append(float(conf))

    # Draw boxes
    for i in range(len(boxes)):
        x, y, width, height = boxes[i]
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, f"Cup {confidences[i]*100:.1f}%", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLOv5 Cup Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
