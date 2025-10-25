from skimage import img_as_ubyte
import cv2

def skimage_to_cv (image):
    if image.ndim == 2:
        return img_as_ubyte(image)
    else:
        return cv2.cvtColor(img_as_ubyte(image), cv2.COLOR_RGB2BGR)