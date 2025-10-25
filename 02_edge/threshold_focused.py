# Import Libraries
from skimage import data, filters, feature, io, color
import matplotlib.pyplot as plt
import numpy as np

# Define Pictures
img_1 = data.camera()
img_2 = color.rgb2gray(io.imread('Scifo.jpg'))

# Sobel Edge Detection
img_1_sobel = filters.sobel(img_1)
img_2_sobel = filters.sobel(img_2)

# Sobel Edge Detection with Custom Threshold 0.05
img_1_sobel_threshed005 = filters.sobel(img_1) > 0.05
img_2_sobel_threshed005 = filters.sobel(img_2) > 0.05

# Sobel Edge Detection with Custom Threshold 0.1
img_1_sobel_threshed010 = filters.sobel(img_1) > 0.1
img_2_sobel_threshed010 = filters.sobel(img_2) > 0.1

# Sobel Edge Detection with Custom Threshold 0.2
img_1_sobel_threshed020 = filters.sobel(img_1) > 0.2
img_2_sobel_threshed020 = filters.sobel(img_2) > 0.2

# Define Picture Grid
fig, axes = plt.subplots(2, 5, figsize=(15, 9))

# Plotting Pictures to the Grid
axes[0,0].imshow(img_1, cmap='gray')
axes[0,0].set_title('Camera - Normal')
axes[0,0].axis('off')

axes[1,0].imshow(img_2, cmap='gray')
axes[1,0].set_title('Scifo - Normal')
axes[1,0].axis('off')

axes[0,1].imshow(img_1_sobel, cmap='gray')
axes[0,1].set_title('Camera - Sobel Edge Detection')
axes[0,1].axis('off')

axes[1,1].imshow(img_2_sobel, cmap='gray')
axes[1,1].set_title('Scifo - Sobel Edge Detection')
axes[1,1].axis('off')

axes[0,2].imshow(img_1_sobel_threshed005, cmap='gray')
axes[0,2].set_title('Camera - Sobel 0.05')
axes[0,2].axis('off')

axes[1,2].imshow(img_2_sobel_threshed005, cmap='gray')
axes[1,2].set_title('Scifo - Sobel 0.05')
axes[1,2].axis('off')

axes[0,3].imshow(img_1_sobel_threshed010, cmap='gray')
axes[0,3].set_title('Camera - Sobel 0.1')
axes[0,3].axis('off')

axes[1,3].imshow(img_2_sobel_threshed010, cmap='gray')
axes[1,3].set_title('Scifo - Sobel 0.1')
axes[1,3].axis('off')

axes[0,4].imshow(img_1_sobel_threshed020, cmap='gray')
axes[0,4].set_title('Camera - Sobel 0.2')
axes[0,4].axis('off')

axes[1,4].imshow(img_2_sobel_threshed020, cmap='gray')
axes[1,4].set_title('Scifo - Sobel 0.2')
axes[1,4].axis('off')

# Show the Pyplot
plt.tight_layout()
plt.show()