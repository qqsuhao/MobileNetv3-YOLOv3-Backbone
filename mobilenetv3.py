# -*- coding:utf8 -*-
# @TIME     : 2022/1/11 15:55
# @Author   : Hao Su
# @File     : mobilenetv3.py


"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""

import sys
import torch
import torch.nn as nn
import math
import struct
import time
import torchvision


__all__ = ['mobilenetv3_large', 'mobilenetv3_small']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        y = self.sigmoid(x)
        return x * y


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False, ),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        y = self.conv(x)
        if self.identity:
            return x + y
        else:
            return y


class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups):
        super(CBL, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size,stride=1, padding=pad,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, exp_size, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)          # 为了yolo可以使用FPN结构
        # building last several layers
        # self.conv = nn.Sequential(
        #     conv_1x1_bn(input_channel, _make_divisible(exp_size * width_mult, 8)),
        #     SELayer(_make_divisible(exp_size * width_mult, 8)) if mode == 'small' else nn.Sequential()
        # )
        # self.avgpool = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     h_swish()
        # )
        # output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        # self.classifier = nn.Sequential(
        #     nn.Linear(_make_divisible(exp_size * width_mult, 8), output_channel),
        #     nn.BatchNorm1d(output_channel) if mode == 'small' else nn.Sequential(),
        #     h_swish(),
        #     nn.Linear(output_channel, num_classes),
        #     nn.BatchNorm1d(num_classes) if mode == 'small' else nn.Sequential(),
        #     h_swish() if mode == 'small' else nn.Sequential()
        # )
        self.yolo1 = nn.Sequential(
            CBL(96, 96, 5, 96),
            CBL(96, 128, 1, 1),
            CBL(128, 128, 5, 128),
        )
        self.yolo2 = nn.Sequential(
            CBL(96+288, 96, 1, 1),
            CBL(96, 96, 5, 96),
            CBL(96, 96, 1, 1),
            CBL(96, 96, 5, 96),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.mid_feature = {}           #! 使用字典解决多GPU训练下的hook不在同一个gpu上的问题

        self._initialize_weights()


    def layer_hook(self, _, fea_in, fea_out):
        self.mid_feature[fea_in[0].device].append(fea_out)

    def forward(self, inputs):

        # FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        # self.mid_feature = FloatTensor
        # mid_feature.to(x.device)
        self.mid_feature[inputs.device] = []
        for index_i, (name, module) in enumerate(self.named_modules()):
            if index_i == 161:
                hook = module.register_forward_hook(hook=self.layer_hook)
                break

        x = self.features(inputs)
        hook.remove()

        ## YOLO
        yolo1 = self.yolo1(x)
        x_upsample = self.upsample(x)
        yolo2 = self.yolo2(torch.cat([x_upsample, self.mid_feature[inputs.device][0]], dim=1))
        # x = self.conv(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return yolo1, yolo2


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _load_pth(self, pthfile, device):
        device = self.cuda()
        pretrained = torch.load(pthfile, map_location=torch.device('cuda'))



def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, NL, s
        [3,  16,  16, 0, 0, 1],
        [3,  64,  24, 0, 0, 2],
        [3,  72,  24, 0, 0, 1],
        [5,  72,  40, 1, 0, 2],
        [5, 120,  40, 1, 0, 1],
        [5, 120,  40, 1, 0, 1],
        [3, 240,  80, 0, 1, 2],
        [3, 200,  80, 0, 1, 1],
        [3, 184,  80, 0, 1, 1],
        [3, 184,  80, 0, 1, 1],
        [3, 480, 112, 1, 1, 1],
        [3, 672, 112, 1, 1, 1],
        [5, 672, 160, 1, 1, 1],
        [5, 672, 160, 1, 1, 2],
        [5, 960, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, NL, s
        [3,  16,  16, 1, 0, 2],
        [3,  72,  24, 0, 0, 2],
        [3,  88,  24, 0, 0, 1],
        [5,  96,  40, 1, 1, 2],
        [5, 240,  40, 1, 1, 1],
        [5, 240,  40, 1, 1, 1],
        [5, 120,  48, 1, 1, 1],
        [5, 144,  48, 1, 1, 1],
        [5, 288,  96, 1, 1, 2],
        [5, 576,  96, 1, 1, 1],
        [5, 576,  96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)


# net_large = mobilenetv3_large()
# net_large.eval()
# net_large.to('cuda:0')
# #net_small = mobilenetv3_small()
# #net_small.eval()
# #net_small.to('cuda:0')
# state_dict = torch.load(('pretrained/mobilenetv3-large-657e7b3d.pth'))
# state_dict["classifier.0.weight"] = state_dict["classifier.1.weight"]
# del state_dict["classifier.1.weight"]
# state_dict["classifier.0.bias"] = state_dict["classifier.1.bias"]
# del state_dict["classifier.1.bias"]
# state_dict["classifier.3.weight"] = state_dict["classifier.5.weight"]
# state_dict["classifier.3.bias"] = state_dict["classifier.5.bias"]
# del state_dict["classifier.5.weight"]
# del state_dict["classifier.5.bias"]
# net_large.load_state_dict(state_dict)
# #net_small.load_state_dict(torch.load('pretrained/mobilenetv3-small-c7eb32fe.pth'))
# #f = open("mbv3_small.wts", "w")
# f = open("mbv3_large.wts", "w")
# f.write("{}\n".format(len(net_large.state_dict().keys())))
# for k, v in net_large.state_dict().items():
#     vr = v.reshape(-1).cpu().numpy()
#     f.write("{} {}".format(k,len(vr)))
#     for vv in vr:
#         f.write(" ")
#         f.write(struct.pack(">f", float(vv)).hex())
#     f.write("\n")
# x = torch.ones(1,3,224,224).to('cuda:0')
# #print(net_small)
# for i in range(10):
#     s = time.time()
#     y = net_large(x)
#     print("time:",time.time() -s)
# print(y)

if __name__ == "__main__":
    from torchsummary import summary
    import os, sys
    os.chdir(sys.path[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mobilenetv3_small().to(device)
    model.load_state_dict(torch.load('mobilenetv3-small-c7eb32fe.pth'), strict=False)
    model = nn.DataParallel(model)

    x = torch.ones(2, 3, 320, 320).to(device)
    x1, x2 = model(x)
    print(x1.size(), x2.size())

    torch.save(model.module.state_dict(), "mobilenetv3-small.pth")