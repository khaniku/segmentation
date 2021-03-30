"""A small Unet-like zoo"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
import torch.nn.functional as F

from src.models.layers import ConvBnRelu, UBlock, conv1x1, UBlockCbam, CBAM, InputTransition, DownTransition, UpTransition, OutputTransition


# class VNet(nn.Module):
#     # the number of convolutions in each layer corresponds
#     # to what is in the actual prototxt, not the intent
#     def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0,
#                  **kwargs):
#         super(VNet, self).__init__()

#         features = [width * 2 ** i for i in range(4)]
#         print(features)

#         self.input_layer = InputTransition(inplanes, features[0], norm_layer)
#         self.down_layer1 = DownTransition(features[0], 1, norm_layer)
#         self.down_layer2 = DownTransition(features[1], 2, norm_layer)
#         self.down_layer3 = DownTransition(features[2], 3, norm_layer, dropout=True)
#         self.down_layer4 = DownTransition(features[3], 2, norm_layer, dropout=True)
#         self.up_layer1 = UpTransition(features[3], features[3], 2, norm_layer, dropout=True)
#         self.up_layer2 = UpTransition(features[3], features[2], 2, norm_layer, dropout=True)
#         self.up_layer3 = UpTransition(features[2], features[1], 1, norm_layer)
#         self.up_layer4 = UpTransition(features[1], features[0], 1, norm_layer)
#         self.out_tr = OutputTransition(features[0], num_classes, norm_layer)

#     def forward(self, x):
#         in_lr = self.input_layer(x)
#         down1 = self.down_layer1(in_lr)
#         down2 = self.down_layer2(down1)
#         down3 = self.down_layer3(down2)
#         down4 = self.down_layer4(down3)
#         out = self.up_layer1(down4, down3)
#         out = self.up_layer2(out, down2)
#         out = self.up_layer3(out, down1)
#         out = self.up_layer4(out, in_lr)
#         out = self.out_tr(out)
#         return out


class conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation_func=nn.PReLU):
        """
        + Instantiate modules: conv-relu-norm
        + Assign them as member variables
        """
        super(conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=1)
        self.relu = activation_func()
        # with learnable parameters
        # self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class conv3d_x3(nn.Module):
    """Three serial convs with a residual connection.
    Structure:
        inputs --> ① --> ② --> ③ --> outputs
                   ↓ --> add--> ↑
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(conv3d_x3, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels, kernel_size)
        self.conv_2 = conv3d(out_channels, out_channels, kernel_size)
        self.conv_3 = conv3d(out_channels, out_channels, kernel_size)

    def forward(self, x):
        z_1 = self.conv_1(x)
        z_3 = self.conv_3(self.conv_2(z_1))
        return z_1 + z_3


class deconv3d_x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, activation_func=nn.PReLU):
        super(deconv3d_x3, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, kernel_size, stride)
        self.lhs_conv = conv3d(out_channels // 2, out_channels, kernel_size)
        self.conv_x3 = conv3d_x3(out_channels, out_channels, kernel_size)

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_add = crop(rhs_up, lhs_conv) + lhs_conv
        return self.conv_x3(rhs_add)


def crop(large, small):
    """large / small with shape [batch_size, channels, depth, height, width]"""

    l, s = large.size(), small.size()
    offset = [0, 0, (l[2] - s[2]) // 2, (l[3] - s[3]) // 2, (l[4] - s[4]) // 2]
    return large[..., offset[2]: offset[2] + s[2], offset[3]: offset[3] + s[3], offset[4]: offset[4] + s[4]]


def conv3d_as_pool(in_channels, out_channels, kernel_size=3, stride=2, activation_func=nn.ReLU):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=1),
        activation_func())


def deconv3d_as_up(in_channels, out_channels, kernel_size=3, stride=2, activation_func=nn.ReLU):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride),
        activation_func()
    )


class softmax_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(softmax_out, self).__init__()
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0)
        self.softmax = F.softmax

    def forward(self, x):
        """Output with shape [batch_size, 1, depth, height, width]."""
        # Do NOT add normalize layer, or its values vanish.
        y_conv = self.conv_2(self.conv_1(x))
        # Put channel axis in the last dim for softmax.
        y_perm = y_conv.permute(0, 2, 3, 4, 1).contiguous()
        y_flat = y_perm.view(-1, 2)
        return self.softmax(y_flat)


class VNet(nn.Module):
    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0, **kwargs):
        features = [width * 2 ** i for i in range(4)]

        super(VNet, self).__init__()
        self.conv_1 = conv3d_x3(inplanes, features[0])
        self.pool_1 = conv3d_as_pool(features[0], features[1])
        self.conv_2 = conv3d_x3(features[1], features[1])
        self.pool_2 = conv3d_as_pool(features[1], features[2])
        self.conv_3 = conv3d_x3(features[2], features[2])
        self.pool_3 = conv3d_as_pool(features[2], features[3])
        self.conv_4 = conv3d_x3(features[3], features[3])
        self.pool_4 = conv3d_as_pool(features[3], features[3] * 2)

        self.bottom = conv3d_x3(features[3]*2, features[3]*2)

        self.deconv_4 = deconv3d_x3(features[3] * 2, features[3] * 2)
        self.deconv_3 = deconv3d_x3(features[3], features[2])
        self.deconv_2 = deconv3d_x3(features[2], features[1])
        self.deconv_1 = deconv3d_x3(features[1], features[0])

        self.out = softmax_out(features[0], num_classes)

    def forward(self, x):
        conv_1 = self.conv_1(x)
        pool = self.pool_1(conv_1)
        conv_2 = self.conv_2(pool)
        pool = self.pool_2(conv_2)
        conv_3 = self.conv_3(pool)
        pool = self.pool_3(conv_3)
        bottom = self.bottom(pool)
        deconv = self.deconv_4(conv_4, bottom)
        deconv = self.deconv_3(conv_3, deconv)
        deconv = self.deconv_2(conv_2, deconv)
        deconv = self.deconv_1(conv_1, deconv)
        return self.out(deconv)



class Unet(nn.Module):
    """Almost the most basic U-net.
    """
    name = "Unet"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print(features)

        self.deep_supervision = deep_supervision

        self.encoder1 = UBlock(inplanes, features[0] // 2, features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlock(features[0], features[1] // 2, features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlock(features[1], features[2] // 2, features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlock(features[2], features[3] // 2, features[3], norm_layer, dropout=dropout)

        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0] // 2, norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0] // 2, num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        down1 = self.encoder1(x)
        down2 = self.downsample(down1)
        down2 = self.encoder2(down2)
        down3 = self.downsample(down2)
        down3 = self.encoder3(down3)
        down4 = self.downsample(down3)
        down4 = self.encoder4(down4)

        bottom = self.bottom(down4)
        bottom_2 = self.bottom_2(torch.cat([down4, bottom], dim=1))

        # Decoder

        up3 = self.upsample(bottom_2)
        up3 = self.decoder3(torch.cat([down3, up3], dim=1))
        up2 = self.upsample(up3)
        up2 = self.decoder2(torch.cat([down2, up2], dim=1))
        up1 = self.upsample(up2)
        up1 = self.decoder1(torch.cat([down1, up1], dim=1))

        out = self.outconv(up1)

        if self.deep_supervision:
            deeps = []
            for seg, deep in zip(
                    [bottom, bottom_2, up3, up2],
                    [self.deep_bottom, self.deep_bottom2, self.deep3, self.deep2]):
                deeps.append(deep(seg))
            return out, deeps

        return out


class EquiUnet(Unet):
    """Almost the most basic U-net: all Block have the same size if they are at the same level.
    """
    name = "EquiUnet"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print(features)

        self.deep_supervision = deep_supervision

        self.encoder1 = UBlock(inplanes, features[0], features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlock(features[0], features[1], features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlock(features[1], features[2], features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlock(features[2], features[3], features[3], norm_layer, dropout=dropout)

        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0], norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0], num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()


class Att_EquiUnet(Unet):
    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False,  dropout=0,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print(features)

        self.deep_supervision = deep_supervision

        self.encoder1 = UBlockCbam(inplanes, features[0], features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlockCbam(features[0], features[1], features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlockCbam(features[1], features[2], features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlockCbam(features[2], features[3], features[3], norm_layer, dropout=dropout)

        self.bottom = UBlockCbam(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = nn.Sequential(
            ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout),
            CBAM(features[2], norm_layer=norm_layer)
        )

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0], norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0], num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()