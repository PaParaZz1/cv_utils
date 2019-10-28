import cv2
import numpy as np


def getY(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return yuv[..., 0]


def extract_grad(img):
    return cv2.Laplacian(img, cv2.CV_8U)


def get_edge_mask_np(img):
    '''
        Args:
            img: np.ndarray(H, W, 3) (0-255)(uint8)
        Return:
            y_channel: np.ndarray(H, W) (0 or 255)(uint8)
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img = cv2.GaussianBlur(img, (5, 5), 0)

    y_channel = getY(img)
    count_num = np.argmax(np.bincount(y_channel.reshape(-1)))  # background color mapping
    y_channel[y_channel == 255] = 254
    y_channel[np.abs(y_channel - count_num) < 5] = 255

    y_grad = extract_grad(y_channel)

    y_channel = cv2.equalizeHist(y_channel)
    y_channel = cv2.morphologyEx(y_channel, cv2.MORPH_CLOSE, kernel)

    y_channel = np.where(y_grad > 3, y_channel, 255)

    y_channel = np.where(y_channel < 200, 255, 0)  # binarize
    return y_channel
