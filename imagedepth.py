import cv2
import numpy as np
from matplotlib import pyplot as plt

# Camera parameters (these would be from your calibration)
FOCAL_LENGTH = 700  # Focal length (in pixels)
BASELINE = 0.1  # Baseline distance between the cameras (in meters)

# Load the left and right images
left_image = cv2.imread('images/leftimage.png', cv2.IMREAD_GRAYSCALE)  # Replace with your image paths
right_image = cv2.imread('images/rightimage.png', cv2.IMREAD_GRAYSCALE)  # Replace with your image paths

# Check if images were loaded
if left_image is None or right_image is None:
    print("Error: Could not load images.")
    exit()

right_image = cv2.resize(right_image, (left_image.shape[1], left_image.shape[0]))

plt.figure()
plt.subplot(121)
plt.imshow(left_image)
plt.subplot(122)
plt.imshow(right_image)
plt.show()
# Convert the images to grayscale
# gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
# gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)


# Create the StereoSGBM object (Stereo Block Matching algorithm)

nDispFactor = 7 #adjust this 
minDisparity = 16

stereo = cv2.StereoSGBM_create(minDisparity= minDisparity,
    numDisparities=16*nDispFactor - minDisparity,  # Example: 80, must be divisible by 16
    blockSize=14,  # Example: smaller block size
    P1=8 * 3 * 7**2,
    P2=32 * 3 * 7**2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY)

# Compute the disparity map
disparity = stereo.compute(left_image, right_image).astype(np.float32) / 16.0   
plt.imshow(disparity, 'gray')
plt.colorbar()
plt.show()

# # Normalize the disparity map for better visualization
disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)


cv2.waitKey(0)
cv2.destroyAllWindows()
