# Import Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, io
from skimage_to_cv import skimage_to_cv

# Define pictures and convert to cv2 compatible data
img = [skimage_to_cv(data.camera()) for _ in range(2)]
img_name = ['checkerboard', 'scifo']

img[0] = skimage_to_cv(data.checkerboard())
img[1] = skimage_to_cv(color.rgb2gray(io.imread('Scifo.jpg')))

# Define pattern size
pattern_size = (7, 7)
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

ret, corners = cv2.findChessboardCorners(img[0], pattern_size, None)

print(img_name[0]+":")
if ret:
    objpoints.append(objp)
    imgpoints.append(corners)

    img_corners = cv2.drawChessboardCorners(cv2.cvtColor(img[0], cv2.COLOR_GRAY2BGR), pattern_size, corners, ret)

    # Camera Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img[0].shape[::-1], None, None
    )

    # 3D World Matrices
    print("✅ Calibration successful!")
    print("\n=== Intrinsic Parameters ===")
    print("Camera Matrix (K):\n", mtx)
    print(f"f_x = {mtx[0,0]:.2f}, f_y = {mtx[1,1]:.2f}")
    print(f"c_x = {mtx[0,2]:.2f}, c_y = {mtx[1,2]:.2f}")

    print("\n=== Distortion Coefficients ===")
    print("k1, k2, p1, p2, k3 = ", dist.ravel())

    print("\n=== Extrinsic Parameters (for this image) ===")
    R, _ = cv2.Rodrigues(rvecs[-1])  # convert rotation vector to matrix
    print("Rotation matrix (R):\n", R)
    print("\nTranslation vector (t):\n", tvecs[-1].ravel())
    
    plt.imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
    plt.title("Detected Chessboard Corners")
    plt.axis("off")
    plt.show()
else:
    print("❌ Checkerboard corners not found. Try a real calibration image.")

