#!/opt/homebrew/bin/python3
"""
Name: depth_estimation.py
Purpose: To estimate the depth of an object in an image
"""

__author__ = "Ojas Chaturvedi"
__github__ = "github.com/ojas-chaturvedi"
__license__ = "MIT"


import cv2
import numpy as np

# Download MiDaS model if not done yet
# !wget https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx -P models


# Load MiDaS Model
path_model = "models/"
model_name = "model-small.onnx"
model = cv2.dnn.readNet(path_model + model_name)

if model.empty():
    print("Could not load the neural net! - Check path")

# Use CPU as backend
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# For GPU, comment above and uncomment following:
# model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

clicked_depth = None


def get_depth_value(event, x, y, flags, param):
    global clicked_depth, depth_map, imgWidth, imgHeight

    half_width = imgWidth

    if event == cv2.EVENT_LBUTTONDOWN:
        if 0 <= y < imgHeight:
            if 0 <= x < half_width:  # Left side (original frame)
                print("Clicked on the original frame. No depth data here.")
            elif half_width <= x < 2 * half_width:  # Right side (depth map)
                depth_x = x - half_width  # Adjust x-coordinate for depth map
                clicked_depth = depth_map[y, depth_x]  # Get depth at adjusted location
                print(f"Depth at ({depth_x}, {y}): {clicked_depth}")
        else:
            print(f"Clicked out of bounds: ({x}, {y})")


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

WINDOW_NAME = "Depth Estimation"
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, get_depth_value)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgHeight, imgWidth, _ = img.shape

    # Create Blob from Input Image
    blob = cv2.dnn.blobFromImage(
        img, 1 / 255.0, (256, 256), (123.675, 116.28, 103.53), True, False
    )

    # Set input to the model
    model.setInput(blob)

    # Forward pass
    output = model.forward()
    output = output[0, :, :]
    output = cv2.resize(
        output, (imgWidth, imgHeight)
    )  # Ensure depth map matches frame size

    # Normalize output (Depth Map)
    depth_map = cv2.normalize(
        output, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Convert to color map
    depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

    # Extract numerical depth information
    center_depth = depth_map[imgHeight // 2, imgWidth // 2]  # Depth at center
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)
    avg_depth = np.mean(depth_map)

    # Overlay depth values on the window
    text_overlay = f"Center Depth: {center_depth:.2f} | Min: {min_depth:.2f} | Max: {max_depth:.2f} | Avg: {avg_depth:.2f}"
    cv2.putText(
        depth_map_colored,
        text_overlay,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # Display clicked depth if available
    if clicked_depth is not None:
        cv2.putText(
            depth_map_colored,
            f"Clicked Depth: {clicked_depth:.2f}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # Combine original and depth map
    combined = np.hstack((frame, depth_map_colored))
    cv2.imshow(WINDOW_NAME, combined)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
