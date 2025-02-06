import cv2
import numpy as np 

camera_id = 0

FOCAL_LENGTH = 700  # in pixels (from camera calibration)
BASELINE = 0.05  # Distance between the two cameras in meters


cap = cv2.VideoCapture(camera_id)
stereo = cv2.StereoBM_create(numDisparities=128, blockSize=11)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Get frame dimensions
    height, width, _ = frame.shape

    # Split into left and right images
    left_frame = frame[:, :width // 2]  # Left half
    right_frame = frame[:, width // 2:]  # Right half

    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(gray_left, gray_right)

    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_normalized = np.uint8(disparity_normalized)

    depth_map = np.zeros_like(disparity, dtype=np.float32)
    
    height, width = disparity.shape

    for i in range(height):
        for j in range(width):
            d = disparity[i, j]
            if d > 0: 
                z = (FOCAL_LENGTH * BASELINE) / d
                depth_map[i, j] = z

    # depth_map = np.zeros_like(disparity, dtype=np.float32)
    # valid_pixels = disparity > 0
    # depth_map[valid_pixels] = (FOCAL_LENGTH * BASELINE) / disparity[valid_pixels]

    depth_map_normalized = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    depth_map_normalized = np.uint8(depth_map_normalized)
    depth_colormap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

    # Display both frames
    # cv2.imshow("Left Camera", left_frame)
    # cv2.imshow("Right Camera", right_frame)
    cv2.imshow("Disparity Map", disparity_normalized)
    cv2.imshow("Depth Map", depth_map_normalized)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



