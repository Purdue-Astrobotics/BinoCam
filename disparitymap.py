import cv2
import numpy as np

# Open the stereo camera
camera_id = 0
cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Stereo Block Matching object
stereo = cv2.StereoBM_create(numDisparities=96, blockSize=11)  # Tune these parameters for better results

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Get frame dimensions
    height, width, _ = frame.shape

    # Split into left and right images
    left_frame = frame[:, :width // 2]  
    right_frame = frame[:, width // 2:]  

    # Convert to grayscale
    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # Compute disparity map
    disparity = stereo.compute(gray_left, gray_right)

    # Normalize the disparity for better visualization
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_normalized = np.uint8(disparity_normalized)

    # Applying a colormap for better visualization
    # disparity_colormap = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)  

    # Display the disparity map
    cv2.imshow("Disparity Map", disparity_normalized)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
