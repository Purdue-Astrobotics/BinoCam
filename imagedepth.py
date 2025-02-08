import cv2
import numpy as np
from matplotlib import pyplot as plt

# Camera parameters
FOCAL_LENGTH = 700  # Focal length (in pixels)
BASELINE = 0.1  # Baseline distance between the cameras (in meters)

# Load images
left_image = cv2.imread('images/imageLeft.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('images/imageRight.png', cv2.IMREAD_GRAYSCALE)

if left_image is None or right_image is None:
    print("Error: Could not load images.")
    exit()

right_image = cv2.resize(right_image, (left_image.shape[1], left_image.shape[0]))

# StereoSGBM parameters
nDispFactor = 7
minDisparity = 16
numDisparities = 16 * nDispFactor - minDisparity  # Must be divisible by 16

stereo_left = cv2.StereoSGBM.create(
    minDisparity=minDisparity,
    numDisparities=numDisparities,
    blockSize=14,
    P1=8 * 3 * 7 ** 2,
    P2=32 * 3 * 7 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)  # Create right matcher

# Compute disparity maps
disparity_left = stereo_left.compute(left_image, right_image).astype(np.float32) / 16.0
disparity_right = stereo_right.compute(right_image, left_image).astype(np.float32) / 16.0

# Apply WLS filter
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_left)
wls_filter.setLambda(8000)  # Regularization strength
wls_filter.setSigmaColor(1.5)  # Edge preservation

filtered_disparity = wls_filter.filter(disparity_left, left_image, disparity_map_right=disparity_right)

# Normalize for visualization
filtered_disparity_normalized = cv2.normalize(filtered_disparity, None, 0, 255, cv2.NORM_MINMAX)
filtered_disparity_normalized = np.uint8(filtered_disparity_normalized)

# Show results
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("Raw Disparity Map")
plt.imshow(disparity_left, cmap='gray')
plt.colorbar()

plt.subplot(122)
plt.title("WLS Filtered Disparity Map")
plt.imshow(filtered_disparity_normalized, cmap='gray')
plt.colorbar()

plt.show()
