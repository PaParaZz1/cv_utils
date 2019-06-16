import cv2
import numpy as np
import edge as E


def sharpen_naive(img, strength=0.5):
    edge = E.gradient_laplace(img)
    sharpen_img = edge.astype(np.float32)*strength + img.astype(np.float32)
    return sharpen_img.clip(0, 255).astype(np.uint8)


def unsharpened_mask(img, strength=0.8):
    assert(strength > 0 and strength < 1)

    def constrast(img):
        x = img.astype(np.float32)
        max_val = x.max()
        min_val = x.min()
        x = (x-min_val)/(max_val-min_val)
        x = 3*x**2 - 2*x**3
        x = x*(max_val-min_val) + min_val
        return x.astype(np.uint8)

    threshold = img.max() // 2
    constrast_img = constrast(img)
    low_freq_img = cv2.GaussianBlur(img, (5, 5), 0.4)
    high_freq_img = img - low_freq_img
    high_freq_img = np.abs(high_freq_img)
    index = np.where(high_freq_img > threshold)

    sharpen_img = img.astype(np.float32)
    constrast_img = constrast_img.astype(np.float32)
    sharpen_img[index] = (1-strength)*sharpen_img[index] + (strength)*constrast_img[index]
    print(sharpen_img.mean())
    sharpen_img = sharpen_img.clip(0, 255).astype(np.uint8)
    return sharpen_img
