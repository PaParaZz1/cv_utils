import torch
import nn_module as M


def test_channel_shuffle():
    inputs = torch.ones(3, 3)
    inputs = torch.stack([inputs, inputs*2, inputs*3], dim=0)
    inputs = inputs.unsqueeze(0).repeat(1, 3, 1, 1)
    channel_shuffle = M.ChannelShuffle(group_num=3)
    print(inputs)
    print(inputs.shape)
    output = channel_shuffle(inputs)
    print(output)


if __name__ == "__main__":
    test_channel_shuffle()
