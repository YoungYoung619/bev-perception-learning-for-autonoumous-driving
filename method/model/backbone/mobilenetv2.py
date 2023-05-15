import os
import warnings
import math
import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo

model_urls = {
    "mobilenetv2_0.1": "https://mirrors.tencent.com/repository/generic/model_packages/mobilenet_v2/mobilenetv2_0.1-7d1d638a.pth",  # noqa: E501
    "mobilenetv2_0.25": "https://mirrors.tencent.com/repository/generic/model_packages/mobilenet_v2/mobilenetv2_0.25-b61d2159.pth",  # noqa: E501
    "mobilenetv2_0.35": "https://mirrors.tencent.com/repository/generic/model_packages/mobilenet_v2/mobilenetv2_0.35-b2e15951.pth",
    "mobilenetv2_0.5": "https://mirrors.tencent.com/repository/generic/model_packages/mobilenet_v2/mobilenetv2_0.5-eaa6f9ad.pth",
    "mobilenetv2_1.0": "https://mirrors.tencent.com/repository/generic/model_packages/mobilenet_v2/mobilenetv2_1.0-0c6065bc.pth"
}

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, width_multi=0.35, out_chnl=128, pretrain=True):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = out_chnl
        interverted_residual_setting = \
        [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            # [6, 64, 4, 1],
            [6, 96, 3, 1],
            [6, 160, 2, 1]
        ]
        self.width_multi = width_multi
        # input_channel = int(input_channel * width_multi)
        input_channel = _make_divisible(input_channel * width_multi, 4 if width_multi == 0.1 else 8)
        self.last_channel = int(last_channel * width_multi) if width_multi > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            # output_channel = int(c * width_multi)
            output_channel = _make_divisible(c * width_multi, 4 if width_multi == 0.1 else 8)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))

                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential

        self.features = nn.Sequential(*self.features)

        self._initialize_weights(pretrain)


    def forward(self, x):
        outs = []
        for idx, module in enumerate(self.features):
            x = module(x)
            if idx in [3, 6, 16]:
                outs.append(x)
        return outs

    def _initialize_weights(self, pretrain):
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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        if pretrain:
            url = model_urls["mobilenetv2_{}".format(self.width_multi)]
            if url is not None:
                pretrained_state_dict = model_zoo.load_url(url,
                                                           map_location='cpu' if not torch.cuda.is_available() else 'cuda')
                print("=> loading pretrained model {}".format(url))
                self.load_state_dict(pretrained_state_dict, strict=False)

    # def load_resume_model(self, model):
    #     if (os.path.exists(model)):
    #         net_dict = self.state_dict()
    #         if not torch.cuda.is_available():
    #             pretrain_dict = torch.load(model, map_location='cpu')
    #         else:
    #             pretrain_dict = torch.load(model)
    #         # print(net_dict.keys())
    #         # print(pretrain_dict.keys())
    #         # for idx, (name, data) in enumerate(pretrain_dict.items()):
    #         #     print(idx, name)
    #         # load_dict = {k: v for k, v in list(pretrain_dict.items()) if
    #         #              k in net_dict and 'conv_' not in k and '17' not in k}
    #         load_dict = {k: v for k, v in pretrain_dict.items() if k in net_dict and net_dict[k].size() == v.size()}
    #         print(f'load keys:{load_dict.keys()}')
    #         net_dict.update(load_dict)
    #         self.load_state_dict(net_dict, strict=True)
    #     else:
    #         print("Not pretrain file [%s] found!"%(model))

