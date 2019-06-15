import cv2
import numpy as np


def gradient_sobel(img, use_gauss=False, return_3c=True):
    x = np.copy(img)
    if use_gauss:
        x = cv2.GaussianBlur(x, (3, 3), 0)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(x, cv2.CV_16S, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(x, cv2.CV_16S, 0, 1, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    sobel = 0.5*sobel_x + 0.5*sobel_y
    if return_3c:
        sobel = sobel[:, :, np.newaxis]
        sobel = sobel.repeat(3, axis=2)
    return sobel


def gradient_laplace(img, use_gauss=False, return_3c=True):
    x = np.copy(img)
    if use_gauss:
        x = cv2.GaussianBlur(x, (3, 3), 0)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = cv2.Laplacian(x, cv2.CV_16S, 3)
    x = cv2.convertScaleAbs(x)
    if return_3c:
        x = x[:, :, np.newaxis]
        x = x.repeat(3, axis=2)
    return x


def gradient_canny(img, use_gauss=False, return_3c=True):
    x = np.copy(img)
    if use_gauss:
        x = cv2.GaussianBlur(x, (3, 3), 0)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(x, 50, 150)
    if return_3c:
        canny = canny[:, :, np.newaxis]
        canny = canny.repeat(3, axis=2)
    return canny