# Import Libraries
from skimage import data, filters, io, color, restoration
from skimage.morphology import disk
import matplotlib.pyplot as plt

# Define Pictures
img_1 = data.camera()
img_2 = color.rgb2gray(io.imread('Scifo.jpg'))

# Gaussian Filter
img_1_gaussian = filters.gaussian(img_1, sigma=4)
img_2_gaussian = filters.gaussian(img_2, sigma=4)

# Median Filter
img_1_median = filters.median(img_1, disk(4))
img_2_median = filters.median(img_2, disk(4))

# Mean Filter
img_1_mean = filters.rank.mean(img_1, disk(4))
img_2_mean = filters.rank.mean(img_2, disk(4))

# Bilateral Filter
img_1_bilateral = restoration.denoise_bilateral(img_1)
img_2_bilateral = restoration.denoise_bilateral(img_2)

# Define Picture Grid
fig, axes = plt.subplots(2, 5, figsize=(15, 9))

# Plot Pictures to the Grid
axes[0,0].imshow(img_1, cmap='gray')
axes[0,0].set_title('Camera - Normal')
axes[0,0].axis('off')

axes[1,0].imshow(img_2, cmap='gray')
axes[1,0].set_title('Scifo - Normal')
axes[1,0].axis('off')

axes[0,1].imshow(img_1_gaussian, cmap='gray')
axes[0,1].set_title('Camera - Gaussian Filter')
axes[0,1].axis('off')

axes[1,1].imshow(img_2_gaussian, cmap='gray')
axes[1,1].set_title('Scifo - Gaussian Filter')
axes[1,1].axis('off')

axes[0,2].imshow(img_1_median, cmap='gray')
axes[0,2].set_title('Camera - Median Filter')
axes[0,2].axis('off')

axes[1,2].imshow(img_2_median, cmap='gray')
axes[1,2].set_title('Scifo - Median Filter')
axes[1,2].axis('off')

axes[0,3].imshow(img_1_mean, cmap='gray')
axes[0,3].set_title('Camera - Mean Filter')
axes[0,3].axis('off')

axes[1,3].imshow(img_2_mean, cmap='gray')
axes[1,3].set_title('Scifo - Mean Filter')
axes[1,3].axis('off')

axes[0,4].imshow(img_1_bilateral, cmap='gray')
axes[0,4].set_title('Camera - Bilateral Filter')
axes[0,4].axis('off')

axes[1,4].imshow(img_2_bilateral, cmap='gray')
axes[1,4].set_title('Scifo - Bilateral Filter')
axes[1,4].axis('off')

# Show the Pyplot
plt.tight_layout()
plt.show()