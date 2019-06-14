import cv2
from functools import partial


def motion_blur(kernel_path, img):
    kernel_motion_blur = cv2.imread(kernel_path)
    kernel_motion_blur = cv2.resize(kernel_motion_blur, (50, 50))
    kernel_motion_blur = kernel_motion_blur / kernel_motion_blur.sum()
    output = cv2.filter2D(img, -1, kernel_motion_blur)
    return output


motion_blur_func = partial(motion_blur, kernel_path='./kernels/kernel_a.png')
