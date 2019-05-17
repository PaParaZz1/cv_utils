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
    mode_list = ['add_multi', 'gaussian']

    def __init__(self, mode, multi_factor_range):
        assert(mode in self.mode_list)
        self.mode = mode
        self.multi_factor_range = multi_factor_range

    def __call__(self, img, sig_read, sig_shot):
        assert(isinstance(img, np.ndarray))
        x = img.copy()
        assert(len(x.shape) in [2])
        if self.mode == 'gaussian':
            variance = np.ones(x.shape) * sig_read**2
            variance += sig_shot*x
            std = np.sqrt(variance)
            H, W = x.shape
            for h in range(H):
                for w in range(W):
                    x[h, w] += np.random.normal(loc=0, scale=std[h, w])
        elif self.mode == 'add_multi':
            x += np.random.normal(loc=0, scale=sig_read, size=x.shape)
            val = self.multi_factor_range
            multi_factor = np.random.uniform(low=-val, high=val, size=x.shape)
            x *= (1+multi_factor)
        else:
            raise ValueError
        return x


if __name__ == "__main__":
    import cv2
    img = cv2.imread('../jun1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    img /= 255
    handle_gp = GaussianPossionNoise(mode='gaussian', multi_factor_range=0.02)
    handle_awgn = AWGN()
    handle_ps = SaltPepperNoise(p=1e-3, done_p=1)
    img_gp = handle_gp(img, 0.1, 0.02)
    img_awgn = handle_awgn(img, 0.2)
    img_ps = handle_ps(img)
    cv2.imwrite('gp.jpg', img_gp*255)
    cv2.imwrite('awgn.jpg', img_awgn*255)
    cv2.imwrite('ps.jpg', img_ps*255)
