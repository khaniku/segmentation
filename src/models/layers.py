from collections import OrderedDict

import torch
from torch import nn, nn as nn
from torch.nn import functional as F


def default_norm_layer(planes, groups=16):
    groups_ = min(groups, planes)
    if planes % groups_ > 0:
        divisor = 16
        while planes % divisor > 0:
            divisor /= 2
        groups_ = int(planes // divisor)
    return nn.GroupNorm(groups_, planes)


def get_norm_layer(norm_type="group"):
    if "group" in norm_type:
        try:
            grp_nb = int(norm_type.replace("group", ""))
            return lambda planes: default_norm_layer(planes, groups=grp_nb)
        except ValueError as e:
            print(e)
            print('using default group number')
            return default_norm_layer
    elif norm_type == "none":
        return None
    else:
        return lambda x: nn.InstanceNorm3d(x, affine=True)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, bias=True):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class ConvBnRelu(nn.Sequential):

    def __init__(self, inplanes, planes, norm_layer=None, dilation=1, dropout=0):
        if norm_layer is not None:
            super(ConvBnRelu, self).__init__(
                OrderedDict(
                    [
                        ('conv', conv3x3(inplanes, planes, dilation=dilation)),
                        ('bn', norm_layer(planes)),
                        ('relu', nn.ReLU(inplace=True)),
                        ('dropout', nn.Dropout(p=dropout)),
                    ]
                )
            )
        else:
            super(ConvBnRelu, self).__init__(
                OrderedDict(
                    [
                        ('conv', conv3x3(inplanes, planes, dilation=dilation, bias=True)),
                        ('relu', nn.ReLU(inplace=True)),
                        ('dropout', nn.Dropout(p=dropout)),
                    ]
                )
            )


class UBlock(nn.Sequential):
    """Unet mainstream downblock.
    """

    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(UBlock, self).__init__(
            OrderedDict(
                [
                    ('ConvBnRelu1', ConvBnRelu(inplanes, midplanes, norm_layer, dilation=dilation[0], dropout=dropout)),
                    (
                        'ConvBnRelu2',
                        ConvBnRelu(midplanes, outplanes, norm_layer, dilation=dilation[1], dropout=dropout)),
                ])
        )


class UBlockCbam(nn.Sequential):
    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(UBlockCbam, self).__init__(
            OrderedDict(
                [
                    ('UBlock', UBlock(inplanes, midplanes, outplanes, norm_layer, dilation=dilation, dropout=dropout)),
                    ('CBAM', CBAM(outplanes, norm_layer=norm_layer)),
                ])
        )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print(UBlock(4, 4))


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 norm_layer=None):
        super(BasicConv, self).__init__()
        bias = False
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self, norm_layer=None):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, norm_layer=norm_layer)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=None, norm_layer=None):
        super(CBAM, self).__init__()
        if pool_types is None:
            pool_types = ['avg', 'max']
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate(norm_layer)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out


def passthrough(x, **kwargs):
    return x

def ELUCons(nchan):
    return nn.PReLU(nchan)

class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)



class LUConv(nn.Module):
    def __init__(self, nchan):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        return x



def _make_nConv(nchan, depth):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, in_planes, outChans):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, outChans, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.relu1 = ELUCons(outChans)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu1(out)
        return out



class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(outChans)
        self.relu2 = ELUCons(outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        x = self.do1(down)
        x = self.ops(x)
        x = self.relu2(torch.add(x, down))
        return x



class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(outChans // 2)
        self.relu2 = ELUCons(outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x, skipx):
        x = self.do1(x)
        skipxdo = self.do2(skipx)
        x = self.relu1(self.bn1(self.up_conv(x)))
        xcat = torch.cat((x, skipxdo), 1)
        x = self.ops(xcat)
        x = self.relu2(torch.add(x, xcat))
        return x



class OutputTransition(nn.Module):
    def __init__(self, inChans, outChans):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=1)
        self.bn1 = ContBatchNorm3d(outChans)
        self.relu1 = ELUCons(outChans)

    def forward(self, x):
      x = self.relu1(self.bn1(self.conv1(x)))
      return x
