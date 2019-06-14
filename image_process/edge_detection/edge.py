import cv2
import numpy as np


def gradient_sobel(img, use_gauss=True):
    x = np.copy(img)
    if use_gauss:
        x = cv2.GaussianBlur(x, (3, 3), 0)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(x, cv2.CV_16S, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(x, cv2.CV_16S, 0, 1, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    return 0.5*sobel_x + 0.5*sobel_y


def gradinet_laplace(img, use_gauss=True):
    x = np.copy(img)
    if use_gauss:
        x = cv2.GaussianBlur(x, (3, 3), 0)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = cv2.Laplacian(x, cv2.CV_16S, 3)
    x = cv2.convertScaleAbs(x)
    return x


def gradient_canny(img, use_gauss=True):
    x = np.copy(img)
    if use_gauss:
        x = cv2.GaussianBlur(x, (3, 3), 0)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(x, 50, 150)
    return canny
