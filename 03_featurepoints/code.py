# Import Libraries
from skimage import data, color, feature, io, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Define Pictures
img_1 = data.camera()
img_2 = color.rgb2gray(io.imread('Scifo.jpg'))

# Convert skimage pictures to cv2
img_1_cv = img_as_ubyte(img_1)
img_2_cv = img_as_ubyte(img_2)

# Feature Points Detection using Harris
coords_1_harris = feature.corner_peaks(feature.corner_harris(img_1), min_distance=15)
coords_2_harris = feature.corner_peaks(feature.corner_harris(img_2), min_distance=15)

# Feature Points Detection using SIFT
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img_1_cv, None)
kp2, des2 = sift.detectAndCompute(img_2_cv, None)

coords_1_sift = np.array([k.pt for k in kp1])
coords_2_sift = np.array([k.pt for k in kp2])

response_1_sift = np.array([k.response for k in kp1])
response_2_sift = np.array([k.response for k in kp2])

# Feature Points Detection using FAST
coords_1_fast = feature.corner_peaks(feature.corner_fast(img_1, n=12, threshold=0.2), min_distance=5)
coords_2_fast = feature.corner_peaks(feature.corner_fast(img_2, n=12, threshold=0.2), min_distance=5)

# Feature Counts
print("=== Feature Counts ===")
print(f"Camera - Harris: {len(coords_1_harris)} points")
print(f"Camera - SIFT:   {len(coords_1_sift)} points")
print(f"Camera - FAST:   {len(coords_1_fast)} points")
print()
print(f"Scifo - Harris:  {len(coords_2_harris)} points")
print(f"Scifo - SIFT:    {len(coords_2_sift)} points")
print(f"Scifo - FAST:    {len(coords_2_fast)} points")

# Patch Responses (SIFT only)
print("\n=== Sample SIFT Responses ===")
print("Camera:", np.round(response_1_sift[:10], 4))
print("Scifo :", np.round(response_2_sift[:10], 4))

# Define the picture grid
fig, axes = plt.subplots(2, 4, figsize=(19, 12))

# Plotting Pictures to the grid
axes[0,0].imshow(img_1, cmap='gray')
axes[0,0].set_title('Camera - Normal')

axes[1,0].imshow(img_2, cmap='gray')
axes[1,0].set_title('Scifo - Normal')

axes[0,1].imshow(img_1, cmap='gray')
axes[0,1].plot(coords_1_harris[:, 1], coords_1_harris[:, 0], 'r.', markersize=5)
axes[0,1].set_title('Camera - Harris Corners')

axes[1,1].imshow(img_2, cmap='gray')
axes[1,1].plot(coords_2_harris[:, 1], coords_2_harris[:, 0], 'r.', markersize=5)
axes[1,1].set_title('Scifo - Harris Corners')

axes[0,2].imshow(img_1, cmap='gray')
axes[0,2].plot(coords_1_sift[:, 0], coords_1_sift[:, 1], 'r.', markersize=5)
axes[0,2].set_title('Camera - Sift Corners')

axes[1,2].imshow(img_2, cmap='gray')
axes[1,2].plot(coords_2_sift[:, 0], coords_2_sift[:, 1], 'r.', markersize=5)
axes[1,2].set_title('Scifo - Sift Corners')

axes[0,3].imshow(img_1, cmap='gray')
axes[0,3].plot(coords_1_fast[:, 1], coords_1_fast[:, 0], 'r.', markersize=5)
axes[0,3].set_title('Camera - Fast Corners')

axes[1,3].imshow(img_2, cmap='gray')
axes[1,3].plot(coords_2_fast[:, 1], coords_2_fast[:, 0], 'r.', markersize=5)
axes[1,3].set_title('Scifo - Fast Corners')

# Show the Pyplot
plt.show()