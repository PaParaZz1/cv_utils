import numpy as np
import cv2


class TransGenerator(object):
    def __init__(self):
        self.mode_list = ['fft', 'ifft', 'dct', 'idct']
        self.mode_dict = {'fft': self.fft,
                          'ifft': self.ifft,
                          'dct': self.dct,
                          'idct': self.idct}

    def fft(self, x, viz=True):
        fft_result = np.fft.fft2(x)
        if viz:
            fft_result = self.viz_fft(fft_result)
        return fft_result

    def ifft(self, x):
        ifft_result = np.fft.ifft2(x)
        return np.real(ifft_result)

    def dct(self, x, viz=True):
        dct_result = cv2.dct(x)
        if viz:
            dct_result = self.viz_dct(dct_result)
        return dct_result

    def idct(self, x):
        idct_result = cv2.idct(x)
        return idct_result

    def transform(self, x, mode, **kwargs):
        func = self.mode_dict[mode]
        return func(x, **kwargs)

    def viz_fft(self, x):
        x = np.fft.fftshift(x)
        x = np.log(1 + np.abs(x))
        x = (x - x.min()) / (x.max() - x.min()+1e-8)
        return x

    def viz_dct(self, x):
        x = np.log(1 + np.abs(x))
        x = (x - x.min()) / (x.max() - x.min()+1e-8)
        return x

    def __call__(self, x, mode):
        assert(mode in self.mode_list)
        assert(isinstance(x, np.ndarray))
        assert(len(x.shape) in [2, 3])
        if len(x.shape) == 3:
            assert(x.shape[2] == 3)
            out = np.zeros(x.shape).astype(np.complex64)
            out[:, :, 0] = self.transform(x[:, :, 0], mode=mode)
        else:
            out = self.transform(x, mode=mode)
        return out
