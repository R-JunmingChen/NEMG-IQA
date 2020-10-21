import torch
import torch.nn as nn
import numpy as np
from util import Context
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

config = Context().get_config()
logger = Context().get_logger()


def get_log_diff_fn_2(exponent=0.2):
    def log_diff_fn(in_a, in_b):
        diff = torch.abs(in_a - in_b)
        val = torch.pow(diff, exponent)
        return val

    return log_diff_fn


def double_value(x):
    return x*x;

def identity_mutiply(x,y):
    indentity=x
    out=x*y
    out=torch.cat([out,indentity],dim=1)
    return out

def identity_add(x,y):
    indentity=x
    out=x+y
    out=torch.cat([out,indentity],dim=1)
    return out

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=6):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
          nn.Linear(channel, int(channel/reduction), bias=False),
          nn.ReLU(inplace=True),
          nn.Linear(int(channel/reduction), channel, bias=False),
          nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).reshape(b, c)
        y = self.fc(y).reshape(b, c, 1, 1)
        return x * y.expand_as(x)

class ConcatSEBlock(nn.Module):
    def __init__(self, channel, reduction=6):
        super(ConcatSEBlock, self).__init__()
        self.se_block = SEBlock(channel,reduction)
    def forward(self, x,concat_list):
        out = x
        if isinstance(concat_list, list):
            for ele in concat_list:
                out = torch.cat([out, ele], dim=1)
        else:
            out = torch.cat([out, concat_list], dim=1)
        out = self.se_block(out)
        return out

class MutiplyConvTrans(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=1, padding=1, bias=False, groups=1):
        super(MutiplyConvTrans, self).__init__()
        self.convtrans = nn.ConvTranspose2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                                            padding=padding, groups=groups, bias=bias)
    def forward(self, x, concat_list):
        out = x
        if isinstance(concat_list, list):
            for ele in concat_list:
                out= out*ele
        else:
            out = out*concat_list
        out = self.convtrans(out)
        return out


class MutiplyConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=1, padding=1, bias=False, groups=1):
        super(MutiplyConv, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=groups, bias=bias)
    def forward(self, x, concat_list):
        out = x
        if isinstance(concat_list, list):
            for ele in concat_list:
                out= out*ele
        else:
            out = out*concat_list
        out = self.convtrans(out)
        return out



class ConcatConvTrans(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=1, padding=1, bias=False, groups=1):
        super(ConcatConvTrans, self).__init__()
        self.convtrans = nn.ConvTranspose2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                                            padding=padding, groups=groups, bias=bias)

    def forward(self, x, concat_list):
        out = x
        if isinstance(concat_list, list):
            for ele in concat_list:
                out = torch.cat([out, ele], dim=1)
        else:
            out = torch.cat([out, concat_list], dim=1)
        out = self.convtrans(out)
        return out


class ConcatConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=1, padding=1, bias=False, groups=1):
        super(ConcatConv, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=groups, bias=bias)

    def forward(self, x, concat_list):
        out = x
        if isinstance(concat_list, list):
            for ele in concat_list:
                out = torch.cat([out, ele], dim=1)
        else:
            out = torch.cat([out, concat_list], dim=1)
        out = self.conv(out)
        return out


class UpSampleFilter(nn.Module):
    def __init__(self):
        super(UpSampleFilter, self).__init__()
        upsample_filter = nn.ConvTranspose2d(1, 1, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2),
                                             output_padding=(1, 1), bias=False)

        k = np.float32([1, 4, 6, 4, 1])
        k = np.outer(k, k)
        k5x5 = (k / k.sum()).reshape((1, 1, 5, 5))
        k5x5 *= 4
        upsample_weight = torch.from_numpy(k5x5)
        upsample_filter.weight = torch.nn.Parameter(upsample_weight)
        upsample_filter.weight.requires_grad = False

        self.filter = upsample_filter

    def forward(self, x):
        if x.shape[1] > 1:
            batch_size, channel, height, width = x.shape
            out = self.filter(x.reshape(batch_size * channel, 1, height, width))

            height, width = out.shape[2], out.shape[3]
            out = out.reshape(batch_size, channel, height, width)
        else:
            out = self.filter(x)
        return out


class DownSampleFilter(nn.Module):
    def __init__(self):
        super(DownSampleFilter, self).__init__()
        downsample_filter = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))

        k = np.float32([1, 4, 6, 4, 1])
        k = np.outer(k, k)
        k5x5 = (k / k.sum()).reshape((1, 1, 5, 5))
        downsample_weight = torch.from_numpy(k5x5)
        downsample_filter.weight = torch.nn.Parameter(downsample_weight)
        downsample_filter.bias.data.fill_(0)
        downsample_filter.weight.requires_grad = False
        downsample_filter.bias.requires_grad = False

        self.filter = downsample_filter

    def forward(self, x):
        if x.shape[1] > 1:
            batch_size, channel, height, width = x.shape
            out = self.filter(x.reshape(batch_size * channel, 1, height, width))

            height, width = out.shape[2], out.shape[3]
            out = out.reshape(batch_size, channel, height, width)
        else:
            out = self.filter(x)
        return out


class SobelX(nn.Module):
    def __init__(self):
        super(SobelX, self).__init__()
        sobel_x_filter = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=1, padding=(1, 1))

        sobel_x_val = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                               dtype='float32').reshape((1, 1, 3, 3))
        sobel_x_filter_weight = torch.from_numpy(sobel_x_val)
        sobel_x_filter.weight = torch.nn.Parameter(sobel_x_filter_weight)
        sobel_x_filter.bias.data.fill_(0)
        sobel_x_filter.weight.requires_grad = False
        sobel_x_filter.bias.requires_grad = False

        self.filter = sobel_x_filter

    def forward(self, x):
        if x.shape[1] > 1:
            batch_size, channel, height, width = x.shape
            out = self.filter(x.reshape(batch_size * channel, 1, height, width))

            height, width = out.shape[2], out.shape[3]
            out = out.reshape(batch_size, channel, height, width)
        else:
            out = self.filter(x)
        return out


class SobelY(nn.Module):
    def __init__(self):
        super(SobelY, self).__init__()
        sobel_y_filter = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=1, padding=(1, 1))

        sobel_y_val = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                               dtype='float32').reshape((1, 1, 3, 3))
        sobel_y_filter_weight = torch.from_numpy(sobel_y_val)
        sobel_y_filter.weight = torch.nn.Parameter(sobel_y_filter_weight)
        sobel_y_filter.bias.data.fill_(0)
        sobel_y_filter.weight.requires_grad = False
        sobel_y_filter.bias.requires_grad = False
        self.filter = sobel_y_filter

    def forward(self, x):
        if x.shape[1] > 1:
            batch_size, channel, height, width = x.shape
            out = self.filter(x.reshape(batch_size * channel, 1, height, width))

            height, width = out.shape[2], out.shape[3]
            out = out.reshape(batch_size, channel, height, width)
        else:
            out = self.filter(x)
        return out
def shave_border( feat_map,ign=4):
    if ign > 0:
        return feat_map[:, :, ign:-ign, ign:-ign]
    else:
        return feat_map



