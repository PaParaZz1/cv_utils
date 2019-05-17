import numpy as np
import torch
import torch.nn.functional as F
import cv2
import noise as N
import transform as T
import denoise as D


def white_balance(img, r_gain, b_gain):
    img[::2, ::2] *= r_gain
    img[1::2, 1::2] *= b_gain
    return img


def demosaic_Malvar2004(img):
    H, W = img.shape
    demosaic = np.zeros((H,W,3))
    filter = np.stack((np.array([
        [0, 0,-1, 0, 0],
        [0, 0, 2, 0, 0],
        [-1,2, 4, 2,-1],
        [0, 0, 2, 0, 0],
        [0, 0,-1, 0, 0],
        ]),np.array([
        [0, 0,.5, 0, 0],
        [0,-1, 0,-1, 0],
        [-1,4, 5, 4,-1],
        [0,-1, 0,-1, 0],
        [0, 0,.5, 0, 0],
        ]),np.array([
        [0, 0,-1, 0, 0],
        [0,-1, 4,-1, 0],
        [.5,0, 5, 0,.5],
        [0,-1, 4,-1, 0],
        [0, 0,-1, 0, 0],
        ]),np.array([
        [0,   0,-1.5, 0,   0],
        [0,   2,  0,  2,   0],
        [-1.5,0,  6,  0,-1.5],
        [0,   2,  0,  2,   0],
        [0,   0,-1.5, 0,   0],
        ])
    )).reshape((4, 1, 5, 5))
    img = torch.from_numpy(img).view(1, 1, H, W).cuda().float()
    kernel = torch.from_numpy(filter).cuda().float()
    kernel /= 8
    ft = F.conv2d(img, kernel, padding=2).view(4, H, W).cpu().numpy()
    img = img.view(H, W).cpu().numpy()
    demosaic[::2, ::2, 2] = img[::2, ::2] # r00
    demosaic[::2, 1::2, 2] = ft[1, ::2, 1::2] # r01
    demosaic[1::2, ::2, 2] = ft[2, 1::2, ::2] # r10
    demosaic[1::2, 1::2, 2] = ft[3, 1::2, 1::2] # r11
    demosaic[::2, ::2, 1] = ft[0, ::2, ::2] # g00
    demosaic[::2, 1::2, 1] = img[::2, 1::2] # g01
    demosaic[1::2, ::2, 1] = img[1::2, ::2] # g10
    demosaic[1::2, 1::2, 1] = ft[0, 1::2, 1::2] # g11
    demosaic[::2, ::2, 0] = ft[3, ::2, ::2] # b00
    demosaic[::2, 1::2, 0] = ft[2, ::2, 1::2] # b01
    demosaic[1::2, ::2, 0] = ft[1, 1::2, ::2] # b10
    demosaic[1::2, 1::2, 0] = img[1::2, 1::2] # b11
    return demosaic[:,:,::-1].copy()


def gamma_transform(x, gamma = 1./2.4):
    if (gamma != 1.):
        x = torch.from_numpy(x).float().cuda()
        b = .0031308
        a = 1./(1./(b**gamma*(1.-gamma))-1.)
        k0 = (1+a)*gamma*b**(gamma-1.)
        srgb = torch.where(x < b, k0 * x, (1+a) * torch.pow(torch.max(x, torch.ones_like(x).cuda() * b), gamma) - a)
        srgb = torch.where(x > 1, (1+a) * gamma * (x - 1) + 1, srgb)
        x = srgb.cpu().numpy()
    return x


def post_process(img):
    assert(isinstance(img, np.ndarray))
    assert(len(img.shape) == 2)
    r_gain = 1.7933
    b_gain = 1.6358
    img_wb = white_balance(img, r_gain, b_gain)
    img_demosaic = demosaic_Malvar2004(img_wb)
    img_gamma = gamma_transform(img_demosaic)
    img_gamma = cv2.cvtColor(img_gamma, cv2.COLOR_RGB2BGR)
    return img_gamma


def pipeline():
    handle_awgn = N.AWGN()
    handle_gp = N.GaussianPossionNoise('gaussian', 0.02)
    handle_ps = N.SaltPepperNoise(p=1e-3, done_p=1)
    handle_transform = T.TransGenerator()
    handle_median = D.MedianFilter(kernel_size=(3, 3))
    handle_gauss = D.GaussFilter(kernel_size=(3, 3), sigma=1.0)
    handle_NLM = D.NonLocalMeanRaw()

    raw = np.load('raw.npy')
    raw_real_noise = np.load('raw_real_noise.npy')
    raw_awgn01 = handle_awgn(raw, sigma=0.002)
    raw_awgn02 = handle_awgn(raw, sigma=0.005)
    raw_gp = handle_gp(raw, sig_read=0.002, sig_shot=0.01)
    raw_ps = handle_ps(raw)
    raw_noise_list = [raw_real_noise, raw_awgn01, raw_awgn02, raw_gp, raw_ps]
    raw_list = [raw, raw_real_noise, raw_awgn01, raw_awgn02, raw_gp, raw_ps]
    func_list = [lambda x: x, handle_median, handle_gauss, handle_NLM]

    def generate_pair(item):
        raw_rgb = post_process(item)
        raw_fft = handle_transform(item, 'fft')
        raw_fft = raw_fft[:, :, np.newaxis].repeat(3, axis=2)
        raw_dct = handle_transform(item, 'dct')
        raw_dct = raw_dct[:, :, np.newaxis].repeat(3, axis=2)
        raws = np.concatenate([raw_rgb, raw_fft, raw_dct], axis=1)
        return raws

    for func in func_list:
        result_list = []
        for i in range(len(raw_list)):
            item = raw_list[i].copy()
            datas = generate_pair(item)
            result_list.append(datas)
            if i != 0:
                item = func(raw_list[i].copy())
                datas = generate_pair(item)
                result_list.append(datas)

        result = (np.concatenate(result_list, axis=0)*255).clip(0, 255)
        cv2.imwrite('{}.jpg'.format(func), result)
        print(func)
    return


if __name__ == "__main__":
    pipeline()
