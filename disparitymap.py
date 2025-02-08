import cv2
import numpy as np

# Open the stereo camera
camera_id = 0
cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Stereo Block Matching (BM) object
num_disparities = 96  # Must be divisible by 16
block_size = 11

stereo_left = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)  # Right matcher for WLS

# Create WLS filter
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_left)
wls_filter.setLambda(8000)  # Controls smoothing strength
wls_filter.setSigmaColor(1.5)  # Preserves edges

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

    # Compute disparity maps
    disparity_left = stereo_left.compute(gray_left, gray_right).astype(np.float32) / 16.0
    disparity_right = stereo_right.compute(gray_right, gray_left).astype(np.float32) / 16.0

    # Apply WLS filter
    filtered_disparity = wls_filter.filter(disparity_left, gray_left, disparity_map_right=disparity_right)

    # Normalize for visualization
    disparity_normalized = cv2.normalize(filtered_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_normalized = np.uint8(disparity_normalized)

    # Display the filtered disparity map
    cv2.imshow("WLS Filtered Disparity Map", disparity_normalized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
