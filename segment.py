#!/opt/homebrew/bin/python3
"""
Name: segment.py
Purpose: Segment objects in real-time using YOLOv8
"""

__author__ = "Ojas Chaturvedi"
__github__ = "github.com/ojas-chaturvedi"
__license__ = "MIT"


import cv2
from ultralytics import YOLO

# Download YOLO model if not done yet (choose one) from this linkL https://huggingface.co/jags/yolov8_model_segmentation-set/tree/main
# !wget https://huggingface.co/jags/yolov8_model_segmentation-set/blob/main/yolov8s-seg.pt -P models

# Load YOLOv8 segmentation model (pretrained)
model = YOLO("models/yolov8s-seg.pt")

# Start video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run segmentation
    results = model(frame)

    # Display output (bounding boxes + masks)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Real-Time Segmentation", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()
