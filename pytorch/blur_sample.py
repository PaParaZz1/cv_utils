import torch
import torch.nn as nn
import torch.nn.functional as F


class BlurPool2d(nn.Module):
    def __init__(self, kernel_size, stride, blur_kernel_learnable=False):
        super(BlurPool2d, self).__init__()
        self.blur_kernel = nn.Parameter(self._get_blur_kernel(kernel_size))
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.kernel_size = kernel_size
        if not blur_kernel_learnable:
            self.blur_kernel.requires_grad_(False)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(-1, H, W).unsqueeze(1)
        x = F.conv2d(x, self.blur_kernel, stride=self.stride, padding=self.padding)
        H, W = x.shape[2:]
        return x.view(B, C, H, W)

    def _get_blur_kernel(self, kernel_size):
        blur_kernel_dict = {
            2: [1, 1],
            3: [1, 2, 1],
            4: [1, 3, 3, 1],
            5: [1, 4, 6, 4, 1],
            6: [1, 5, 10, 10, 5, 1],
            7: [1, 6, 15, 20, 15, 6, 1]
        }
        if kernel_size in blur_kernel_dict.keys():
            blur_kernel_1d = torch.FloatTensor(blur_kernel_dict[kernel_size]).view(-1, 1)
            blur_kernel = torch.matmul(blur_kernel_1d, blur_kernel_1d.t())
            blur_kernel.div_(blur_kernel.sum())
            return blur_kernel.unsqueeze(0).unsqueeze(1)
        else:
            raise ValueError("invalid blur kernel size: {}".format(kernel_size))

    def __repr__(self):
        return 'BlurPool2d(kernel_size=({}, {}), stride=({}, {}), padding=({}, {}))'.format(
                    self.kernel_size, self.kernel_size, self.stride,
                    self.stride, self.padding, self.padding
                )


class MaxBlurPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0, blur_kernel_size=3, blur_kernel_learnable=False, blur_position='after'):
        super(MaxBlurPool2d, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=padding)
        self.blurpool = BlurPool2d(kernel_size=blur_kernel_size, stride=stride, blur_kernel_learnable=blur_kernel_learnable)

        if blur_position == 'after':
            self.layer = [self.maxpool, self.blurpool]
        elif blur_position == 'before':
            self.layer = [self.blurpool, self.maxpool]
        else:
            raise ValueError('invalid blur postion: {}'.format(blur_position))

        self.main = nn.Sequential(self.maxpool, self.blurpool)

    def forward(self, x):
        return self.main(x)


class ConvBlurBlock(nn.Module):
    def __init__(self, *args, stride=2, blur_kernel_size=3, activation=nn.ReLU(), blur_kernel_learnable=False, blur_position='after', **kwargs):
        super(ConvBlurBlock, self).__init__()
        self.conv = nn.Conv2d(*args, **kwargs, stride=1)
        self.activation = activation
        self.blurpool = BlurPool2d(kernel_size=blur_kernel_size, stride=stride, blur_kernel_learnable=blur_kernel_learnable)

        if blur_position == 'after':
            self.layer = [self.conv, self.activation, self.blurpool]
        elif blur_position == 'before':
            self.layer = [self.blurpool, self.conv, self.activation]
        else:
            raise ValueError('invalid blur position: {}'.format(blur_position))
        self.main = nn.Sequential(*self.layer)

    def forward(self, x):
        return self.main(x)


def test():
    m = ConvBlurBlock(3,6,3, padding=1, dilation=1, groups=1, bias=True)
    #m = MaxBlurPool2d(kernel_size=2, stride=2)
    print(m)
    inputs = torch.randn(4, 3, 32, 32)
    output = m(inputs)
    print(output.shape)


if __name__ == "__main__":
    test()
