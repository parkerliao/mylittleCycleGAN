import torch
import numpy as np
from torch.nn import init


def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1 or classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.2)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Norm') != -1:
        init.normal_(m.weight.data, 1, 0.2)


def init_net(net):

    if torch.cuda.is_available():
        net.cuda()

    net.apply(init_weight)

    return net


def set_requires_grad(nets,requires_grad=True):
    if not isinstance(nets,list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def merge_images(sources, targets,batch_size ,k=10):
    _, _, h, w = sources.shape
    row = int(np.sqrt(batch_size))
    merged = np.zeros([3, row*h, row*w*2])
    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row
        merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
        merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
    return merged.transpose(1, 2, 0)