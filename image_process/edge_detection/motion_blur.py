import cv2
from functools import partial


def gauss_blur(img):
    return cv2.GaussianBlur(img, (7,7), 0)


def motion_blur(img, kernel_path):
    kernel_motion_blur = cv2.imread(kernel_path)
    kernel_motion_blur = cv2.resize(kernel_motion_blur, (50, 50))
    kernel_motion_blur = kernel_motion_blur / kernel_motion_blur.sum()
    output = cv2.filter2D(img, -1, kernel_motion_blur*3)
    return output


motion_blur_func_nonlinear = partial(motion_blur, kernel_path='./kernels/kernel_c.png')
motion_blur_func_linear = partial(motion_blur, kernel_path='./kernels/hand_1.png')
