import numpy as np
import cv2
from denoise_network import IDenoiseNet

class BaseFilter(object):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def pad(self, kernel_size, img, mode='reflect'):
        pad_h = kernel_size[0] // 2
        pad_w = kernel_size[1] // 2
        pad_img = np.pad(img, [pad_h, pad_w], mode=mode)
        return pad_img


class MedianFilter(BaseFilter):
    def __init__(self, kernel_size=(3,3)):
        super(MedianFilter, self).__init__(kernel_size)

    def __call__(self, img):
        assert(isinstance(img, np.ndarray))
        assert(len(img.shape) in [2])
        H, W = img.shape
        kh, kw = self.kernel_size[0], self.kernel_size[1]

        output = np.zeros((H, W))
        for i in range(2):
            for j in range(2):
                pad_img = self.pad(self.kernel_size, img[i::2, j::2], mode='reflect')
                for h in range(H//2):
                    for w in range(W//2):
                        output[h*2+i, w*2+j] = np.median(pad_img[h:h+kh, w:w+kw])

        return output


def gauss2D(shape=(3, 3), sigma=1):
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y)/(2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


class GaussFilter(BaseFilter):
    def __init__(self, kernel_size=(3,3), sigma=1.0):
        super(GaussFilter, self).__init__(kernel_size)
        self.kernel = gauss2D(shape=kernel_size, sigma=sigma)

    def __call__(self, img):
        assert(isinstance(img, np.ndarray))
        assert(len(img.shape) in [2])
        H, W = img.shape
        kh, kw = self.kernel_size[0], self.kernel_size[1]
        pad_img = self.pad(self.kernel_size, img, 'reflect')

        output = np.zeros(img.shape)
        for i in range(2):
            for j in range(2):
                pad_img = self.pad(self.kernel_size, img[i::2, j::2], mode='reflect')
                for h in range(H//2):
                    for w in range(W//2):
                        output[h*2+i, w*2+j] = (self.kernel*pad_img[h:h+kh, w:w+kw]).sum()

        return output


class NonLocalMeanRaw(object):
    def __init__(self):
        pass

    def __call__(self, img):
        assert(isinstance(img, np.ndarray))
        assert(len(img.shape) in [2])

        H, W = img.shape
        x = img.copy()*255
        factor = 255.0 / (x.max() - x.min())
        factor *= 0.4
        output = np.zeros(img.shape)
        for i in range(2):
            for j in range(2):
                single_pattern = x[i::2, j::2]
                denoise_ori = cv2.fastNlMeansDenoising(single_pattern.astype(np.uint8), None, factor, 3, 21)  # last 3:filter factor, window size, search size
                denoise_down = cv2.resize(denoise_ori.astype(np.float32), (0, 0), fx=0.25, fy=0.25)

                single_pattern_down = cv2.resize(single_pattern, (0, 0), fx=0.25, fy=0.25)
                down_denoise = cv2.fastNlMeansDenoising(single_pattern_down.astype(np.uint8), None, factor/2.0, 3, 21)

                diff_down = denoise_down - down_denoise
                diff_up = cv2.resize(diff_down, (W//2, H//2))
                denoise_result = denoise_ori - diff_up
                output[i::2, j::2] = denoise_result
        output = output.clip(0, 255)
        return output/255
