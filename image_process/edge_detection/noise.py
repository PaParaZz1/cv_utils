import torch
import numpy as np


class AWGN(object):
    def __init__(self):
        pass

    def __call__(self, x, sigma):
        assert(isinstance(x, np.ndarray))
        assert(sigma > 0)
        output = x + np.random.normal(loc=0.0, scale=sigma, size=x.shape)
        return output


class SaltPepperNoise(object):
    '''
    Args:
        p: probability of add noise on pixel
        done_p: probability of doing add noise operation
    '''
    def __init__(self, p, done_p=0):
        self.p = p
        self.dead_tensor = torch.FloatTensor([-1])
        self.hot_tensor = torch.FloatTensor([2])
        self.dead_point = torch.FloatTensor([0])
        self.hot_point = torch.FloatTensor([1])
        self.done_p = done_p

    def __call__(self, img):
        assert(isinstance(img, torch.Tensor) or isinstance(img, np.ndarray))
        if isinstance(img, np.ndarray):
            x = torch.from_numpy(img.copy())
        else:
            x = img.clone()
        assert(len(x.shape) in [2, 3])

        if np.random.uniform(0, 1) > self.done_p:
            return x.numpy()
        mask = np.random.uniform(0, 1, size=x.shape)
        mask = torch.from_numpy(mask).float()
        mask = torch.where(mask < self.p, self.dead_tensor, mask)
        mask = torch.where(mask > 1-self.p, self.hot_tensor, mask)

        x = torch.where(mask == self.dead_tensor, self.dead_point, x)
        x = torch.where(mask == self.hot_tensor, self.hot_point, x)
        x = x.numpy()
        return x


class GaussianPossionNoise(object):
    def __init__(self, mode, multi_factor_range):
        assert(mode in self.mode_list)
        self.mode = mode
        self.multi_factor_range = multi_factor_range

    def __call__(self, img, sig_read, sig_shot):
        assert(isinstance(img, np.ndarray))
        x = img.copy()
        assert(len(x.shape) in [2])
        variance = np.ones(x.shape) * sig_read**2
        variance += sig_shot*x
        std = np.sqrt(variance)
        H, W = x.shape
        for h in range(H):
            for w in range(W):
                x[h, w] += np.random.normal(loc=0, scale=std[h, w])
        return x

