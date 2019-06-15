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
    def __init__(self, p, done_p=1):
        self.p = p
        self.dead_tensor = torch.FloatTensor([-1])
        self.hot_tensor = torch.FloatTensor([2])
        self.dead_point = torch.FloatTensor([0])
        self.hot_point = torch.FloatTensor([1])
        self.done_p = done_p

    def __call__(self, img):
        assert(isinstance(img, torch.Tensor) or isinstance(img, np.ndarray))
        if isinstance(img, np.ndarray):
            y = torch.from_numpy(img.copy())
        else:
            y = img.clone()
        assert(len(y.shape) in [2, 3])

        if np.random.uniform(0, 1) > self.done_p:
            return y.numpy()
        if len(y.shape) == 2:
            y = y[:, :, np.newaxis]
        for i in range(y.shape[2]):
            x = y[:, :, i]
            mask = np.random.uniform(0, 1, size=x.shape)
            mask = torch.from_numpy(mask).float()
            mask = torch.where(mask < self.p, self.dead_tensor, mask)
            mask = torch.where(mask > 1-self.p, self.hot_tensor, mask)

            x = torch.where(mask == self.dead_tensor, self.dead_point, x)
            x = torch.where(mask == self.hot_tensor, self.hot_point, x)
            y[:, :, i] = x
        y = y.numpy()
        return y


class GaussianPossionNoise(object):
    def __init__(self):
        pass

    def __call__(self, img, sig_read, sig_shot):
        assert(isinstance(img, np.ndarray))
        y = img.copy()
        assert(len(y.shape) in [2, 3])
        if len(y.shape) == 2:
            y = y[:, :, np.newaxis]
        for i in range(y.shape[2]):
            x = y[:, :, i]
            variance = np.ones(x.shape) * sig_read**2
            variance += sig_shot*x
            std = np.sqrt(variance)
            H, W = x.shape
            for h in range(H):
                for w in range(W):
                    x[h, w] += np.random.normal(loc=0, scale=std[h, w])
            y[:, :, i] = x
        return y


def noise_gauss(img):
    handle = AWGN()
    img = img.astype(np.float32) / 255
    img = handle(img, 1e-2)
    img = (img*255).clip(0, 255).astype(np.uint8)
    return img


def noise_gauss_possion(img):
    handle = GaussianPossionNoise()
    img = img.astype(np.float32) / 255
    img = handle(img, 1e-3, 1e-2)
    img = (img*255).clip(0, 255).astype(np.uint8)
    return img


def noise_salt_pepper(img):
    handle = SaltPepperNoise(p=1e-3)
    img = img.astype(np.float32) / 255
    img = handle(img)
    img = (img*255).clip(0, 255).astype(np.uint8)
    return img
