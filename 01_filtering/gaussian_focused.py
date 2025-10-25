# Import Libraries
from skimage import data, filters, io, color, restoration
from skimage.morphology import disk
import matplotlib.pyplot as plt

# Define Pictures
img_1 = data.camera()
img_2 = color.rgb2gray(io.imread('Scifo.jpg'))

# Gaussian Filter
img_1_gaussian2 = filters.gaussian(img_1, sigma=2)
img_2_gaussian2 = filters.gaussian(img_2, sigma=2)

img_1_gaussian4 = filters.gaussian(img_1, sigma=4)
img_2_gaussian4 = filters.gaussian(img_2, sigma=4)

img_1_gaussian6 = filters.gaussian(img_1, sigma=6)
img_2_gaussian6 = filters.gaussian(img_2, sigma=6)

# Define Picture Grid
fig, axes = plt.subplots(2, 4, figsize=(15, 9))

# Plot Pictures to the Grid
axes[0,0].imshow(img_1, cmap='gray')
axes[0,0].set_title('Camera - Normal')
axes[0,0].axis('off')

axes[1,0].imshow(img_2, cmap='gray')
axes[1,0].set_title('Scifo - Normal')
axes[1,0].axis('off')

axes[0,1].imshow(img_1_gaussian2, cmap='gray')
axes[0,1].set_title('Camera - Gaussian Sigma 2')
axes[0,1].axis('off')

axes[1,1].imshow(img_2_gaussian2, cmap='gray')
axes[1,1].set_title('Scifo - Gaussian Sigma 2')
axes[1,1].axis('off')

axes[0,2].imshow(img_1_gaussian4, cmap='gray')
axes[0,2].set_title('Camera - Gaussian Sigma 4')
axes[0,2].axis('off')

axes[1,2].imshow(img_2_gaussian4, cmap='gray')
axes[1,2].set_title('Scifo - Gaussian Sigma 4')
axes[1,2].axis('off')

axes[0,3].imshow(img_1_gaussian6, cmap='gray')
axes[0,3].set_title('Camera - Gaussian Sigma 6')
axes[0,3].axis('off')

axes[1,3].imshow(img_2_gaussian6, cmap='gray')
axes[1,3].set_title('Scifo - Gaussian Sigma 6')
axes[1,3].axis('off')

# Show the Pyplot
plt.tight_layout()
plt.show()