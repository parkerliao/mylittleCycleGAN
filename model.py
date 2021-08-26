import torch
import torch.nn as nn
import utils


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True, ln=False, activation=True):
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    elif ln:
        layers.append(nn.InstanceNorm2d(c_out, affine=True))
    if activation:
        layers.append(nn.LeakyReLU(0.02))
    return nn.Sequential(*layers)


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True, ln=False, activation=True):
    layers = []
    layers.append(nn.ConvTranspose2d(
        c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    elif ln:
        layers.append(nn.InstanceNorm2d(c_out, affine=True))
    if activation:
        layers.append(nn.LeakyReLU(0.02))
    return nn.Sequential(*layers)


class Disc(nn.Module):
    def __init__(self, in_dim, dim=64, bn=True, ln=False):
        super(Disc, self).__init__()
        self.model = nn.Sequential(
            conv(in_dim, dim, 4, bn=False),
            conv(dim, dim*2, 4, bn=bn, ln=ln),
            conv(dim*2, dim*4, 4, bn=bn, ln=ln),
            conv(dim*4, 1, 4, 1, 0, False, False)
        )

    def forward(self, x):
        return self.model(x)


class Gen(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, conv_dim=64, bn=True, ln=False):
        super(Gen, self).__init__()

        self.conv1 = conv(in_dim, conv_dim, 4, bn=bn, ln=ln)
        self.conv2 = conv(conv_dim, conv_dim*2, 4, bn=bn, ln=ln)

        self.residual1 = conv(conv_dim*2, conv_dim*2, 3, 1, 1, bn=bn, ln=ln)
        self.residual2 = conv(conv_dim*2, conv_dim*2, 3, 1, 1, bn=bn, ln=ln)

        self.deconv1 = deconv(conv_dim*2, conv_dim, 4, bn=bn, ln=ln)
        self.deconv2 = deconv(conv_dim, out_dim, 4, bn=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = nn.functional.tanh(x)

        return x


def define_G(in_dim, out_dim, conv_dim, norm_type):

    net = None

    if norm_type == "batch":
        net = Gen(in_dim, out_dim, conv_dim, bn=True, ln=False)
    elif norm_type == "instance":
        net = Gen(in_dim, out_dim, conv_dim, bn=False, ln=True)
    else:
        net = Gen(in_dim, out_dim, conv_dim, bn=False, ln=False)

    utils.init_net(net)

    return net


def define_D(in_dim, conv_dim, norm_type):
    net = None

    if norm_type == "batch":
        net = Disc(in_dim, dim=conv_dim, bn=True, ln=False)
    elif norm_type == "instance":
        net = Disc(in_dim, dim=conv_dim, bn=False, ln=True)
    else:
        net = Disc(in_dim, dim=conv_dim, bn=False, ln=False)

    utils.init_net(net)

    return net
