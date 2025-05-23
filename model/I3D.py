import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):

        (batch, channel, t, h, w) = x.size()
        # ---------------------------------#
        # compute 'same' padding
        # 分别计算维度 t,h以及w的pad
        # ---------------------------------#
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f

        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f

        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f
        # ----------------------#
        # 将三个维度的pad分别表示
        # 出来之后,将pad求出
        # ----------------------#
        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        return super(MaxPool3dSamePadding, self).forward(x)


# --------------------------------------#
# 对于Unit3D这个类的定义
# 我们将Conv3d中的padding设置为0,我们将会
# 根据输入的变化来动态的进行pad
# 对于这个模块我们将其看作2D目标检测中的
# conv+bn+relu
# 即为大结构中一次普通的卷积操作
# --------------------------------------#
class Unit3D(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1), stride=(1, 1, 1),
                 padding=0, activation_fn=F.gelu, use_batch_norm=True, use_bias=False, name='unit_3d'):
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride, padding=self.padding, bias=self._use_bias)
        # ------------------------------#
        # 在该类中为use_batch_norm=True
        # 表明将会使用3d的BatchNorm
        # ------------------------------#
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001)# , momentum=0.1)
            self.bn.eval()


    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # --------------------------------#
        # 在这个类中其具体的实现顺序为
        # 3d卷积 -> BatchNorm3d -> relu
        # --------------------------------#
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


# -----------------------------------#
# 此为对于InceptionModule的定义
# 为多次卷积堆叠而来的结构
# 将会在InceptionI3d使用
# -----------------------------------#
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')

        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=1,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')

        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=1,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')

        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')

        self.name = name

    def forward(self, x):
        # ---------------------------------#
        # 此为根据Inception的结构进行搭建
        # 总共有四个输出,之后将它们堆叠然后输出
        # ---------------------------------#
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))

        return torch.cat([b0, b1, b2, b3], dim=1)

