from torch import nn, Tensor
import argparse
import torch
from typing import Optional, Union, Tuple, List, Any
from torch.nn import functional as F
from typing import Optional
import math

from .normalization import (
    BatchNorm1d,
    BatchNorm2d,
    SyncBatchNorm,
    LayerNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    GroupNorm,
    BatchNorm3d,
    LayerNorm2D,
)

from .activation import (
    ReLU,
    Hardswish,
    Hardsigmoid,
    PReLU,
    LeakyReLU,
    Swish,
    GELU,
    Sigmoid,
    ReLU6,
    Tanh,
)

def module_profile(module, x: Tensor, *args, **kwargs) -> Tuple[Tensor, float, float]:
    """
    Helper function to profile a module.
    .. note::
        Module profiling is for reference only and may contain errors as it solely relies on user implementation to
        compute theoretical FLOPs
    """

    if isinstance(module, nn.Sequential):
        n_macs = n_params = 0.0
        for l in module:
            try:
                x, l_p, l_macs = l.profile_module(x)
                n_macs += l_macs
                n_params += l_p
            except Exception as e:
                print(e, l)
                pass
    else:
        x, n_params, n_macs = module.profile_module(x)
    return x, n_params, n_macs

def get_normalization_layer(
        opts,
        num_features: int,
        norm_type: Optional[str] = None,
        num_groups: Optional[int] = None,
        *args,
        **kwargs
) -> nn.Module:
    """
    Helper function to get normalization layers
    """
    norm_type = (
        opts['normalization']['name']
        if norm_type is None
        else norm_type
    )
    num_groups = (
        opts['normalization']['groups'] if 'groups' in opts['normalization'] else 1
        if num_groups is None
        else num_groups
    )
    momentum = opts['normalization']['momentum']

    norm_layer = None
    norm_type = norm_type.lower() if norm_type is not None else None
    if norm_type in ["batch_norm", "batch_norm_2d"]:
        norm_layer = BatchNorm2d(num_features=num_features, momentum=momentum)
    elif norm_type == "batch_norm_3d":
        return BatchNorm3d(num_features=num_features, momentum=momentum)
    elif norm_type == "batch_norm_1d":
        norm_layer = BatchNorm1d(num_features=num_features, momentum=momentum)
    elif norm_type in ["sync_batch_norm", "sbn"]:
        if torch.cuda.device_count() > 1:
            norm_layer = SyncBatchNorm(num_features=num_features, momentum=momentum)
        else:
            norm_layer = BatchNorm2d(num_features=num_features, momentum=momentum)
    elif norm_type in ["group_norm", "gn"]:
        num_groups = math.gcd(num_features, num_groups)
        norm_layer = GroupNorm(num_channels=num_features, num_groups=num_groups)
    elif norm_type in ["instance_norm", "instance_norm_2d"]:
        norm_layer = InstanceNorm2d(num_features=num_features, momentum=momentum)
    elif norm_type == "instance_norm_1d":
        norm_layer = InstanceNorm1d(num_features=num_features, momentum=momentum)
    elif norm_type in ["layer_norm", "ln"]:
        norm_layer = LayerNorm(num_features)
    elif norm_type in ["layer_norm_2d"]:
        norm_layer = LayerNorm2D(num_features=num_features)
    elif norm_type == "identity":
        norm_layer = Identity()
    else:
        raise NotImplementedError
    return norm_layer


def get_activation_fn(
        act_type: Optional[str] = "relu",
        num_parameters: Optional[int] = -1,
        inplace: Optional[bool] = True,
        negative_slope: Optional[float] = 0.1,
        *args,
        **kwargs
) -> nn.Module:
    """
    Helper function to get activation (or non-linear) function
    """

    if act_type == "relu":
        return ReLU(inplace=False)
    elif act_type == "prelu":
        assert num_parameters >= 1
        return PReLU(num_parameters=num_parameters)
    elif act_type == "leaky_relu":
        return LeakyReLU(negative_slope=negative_slope, inplace=inplace)
    elif act_type == "hard_sigmoid":
        return Hardsigmoid(inplace=inplace)
    elif act_type == "swish":
        return Swish()
    elif act_type == "gelu":
        return GELU()
    elif act_type == "sigmoid":
        return Sigmoid()
    elif act_type == "relu6":
        return ReLU6(inplace=inplace)
    elif act_type == "hard_swish":
        return Hardswish(inplace=inplace)
    elif act_type == "tanh":
        return Tanh()
    else:
        raise NotImplementedError


class BaseLayer(nn.Module):
    """
    Base class for neural network layers
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add layer specific arguments"""
        return parser

    def forward(self, *args, **kwargs) -> Any:
        pass

    def profile_module(self, *args, **kwargs) -> Tuple[Tensor, float, float]:
        raise NotImplementedError

    def __repr__(self):
        return "{}".format(self.__class__.__name__)


class Conv2d(nn.Conv2d):
    """
    Applies a 2D convolution over an input
    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Defaults to 1
        padding (Union[int, Tuple[int, int]]): Padding for convolution. Defaults to 0
        dilation (Union[int, Tuple[int, int]]): Dilation rate for convolution. Default: 1
        groups (Optional[int]): Number of groups in convolution. Default: 1
        bias (bool): Use bias. Default: ``False``
        padding_mode (Optional[str]): Padding mode. Default: ``zeros``
        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``
        act_name (Optional[str]): Use specific activation function. Overrides the one specified in command line args.
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Optional[Union[int, Tuple[int, int]]] = 1,
            padding: Optional[Union[int, Tuple[int, int]]] = 0,
            dilation: Optional[Union[int, Tuple[int, int]]] = 1,
            groups: Optional[int] = 1,
            bias: Optional[bool] = False,
            padding_mode: Optional[str] = "zeros",
            *args,
            **kwargs
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )


class ConvLayer(BaseLayer):
    """
    Applies a 2D convolution over an input
    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Default: 1
        dilation (Union[int, Tuple[int, int]]): Dilation rate for convolution. Default: 1
        groups (Optional[int]): Number of groups in convolution. Default: 1
        bias (Optional[bool]): Use bias. Default: ``False``
        padding_mode (Optional[str]): Padding mode. Default: ``zeros``
        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``
        act_name (Optional[str]): Use specific activation function. Overrides the one specified in command line args.
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    .. note::
        For depth-wise convolution, `groups=C_{in}=C_{out}`.
    """

    def __init__(
            self,
            opts,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Optional[Union[int, Tuple[int, int]]] = 1,
            dilation: Optional[Union[int, Tuple[int, int]]] = 1,
            groups: Optional[int] = 1,
            bias: Optional[bool] = False,
            padding_mode: Optional[str] = "zeros",
            use_norm: Optional[bool] = True,
            use_act: Optional[bool] = True,
            act_name: Optional[str] = None,
            *args,
            **kwargs
    ) -> None:
        super().__init__()

        if use_norm:
            norm_type = opts['normalization']['name']
            if norm_type is not None and norm_type.find("batch") > -1:
                assert not bias, "Do not use bias when using normalization layers."
            elif norm_type is not None and norm_type.find("layer") > -1:
                bias = True
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)
        assert isinstance(dilation, Tuple)

        padding = (
            int((kernel_size[0] - 1) / 2) * dilation[0],
            int((kernel_size[1] - 1) / 2) * dilation[1],
        )

        if in_channels % groups != 0:
            print(
                "Input channels are not divisible by groups. {}%{} != 0 ".format(
                    in_channels, groups
                )
            )
        if out_channels % groups != 0:
            print(
                "Output channels are not divisible by groups. {}%{} != 0 ".format(
                    out_channels, groups
                )
            )

        block = nn.Sequential()

        conv_layer = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        block.add_module(name="conv", module=conv_layer)

        self.norm_name = None
        if use_norm:
            norm_layer = get_normalization_layer(opts=opts, num_features=out_channels)
            block.add_module(name="norm", module=norm_layer)
            self.norm_name = norm_layer.__class__.__name__

        self.act_name = None
        act_type = (
            opts['activation']['name']
            if act_name is None
            else act_name
        )

        if act_type is not None and use_act:
            neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
            inplace = getattr(opts, "model.activation.inplace", False)
            act_layer = get_activation_fn(
                act_type=act_type,
                inplace=inplace,
                negative_slope=neg_slope,
                num_parameters=out_channels,
            )
            block.add_module(name="act", module=act_layer)
            self.act_name = act_layer.__class__.__name__

        self.block = block

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.kernel_size = conv_layer.kernel_size
        self.bias = bias
        self.dilation = dilation

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        cls_name = "{} arguments".format(cls.__name__)
        group = parser.add_argument_group(title=cls_name, description=cls_name)
        group.add_argument(
            "--model.layer.conv-init",
            type=str,
            default="kaiming_normal",
            help="Init type for conv layers",
        )
        parser.add_argument(
            "--model.layer.conv-init-std-dev",
            type=float,
            default=None,
            help="Std deviation for conv layers",
        )
        return parser

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

    def __repr__(self):
        repr_str = self.block[0].__repr__()
        repr_str = repr_str[:-1]

        if self.norm_name is not None:
            repr_str += ", normalization={}".format(self.norm_name)

        if self.act_name is not None:
            repr_str += ", activation={}".format(self.act_name)
        repr_str += ", bias={}".format(self.bias)
        repr_str += ")"
        return repr_str

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        if input.dim() != 4:
            print(
                "Conv2d requires 4-dimensional input (BxCxHxW). Provided input has shape: {}".format(
                    input.size()
                )
            )

        b, in_c, in_h, in_w = input.size()
        assert in_c == self.in_channels, "{}!={}".format(in_c, self.in_channels)

        stride_h, stride_w = self.stride
        groups = self.groups

        out_h = in_h // stride_h
        out_w = in_w // stride_w

        k_h, k_w = self.kernel_size

        # compute MACS
        macs = (k_h * k_w) * (in_c * self.out_channels) * (out_h * out_w) * 1.0
        macs /= groups

        if self.bias:
            macs += self.out_channels * out_h * out_w

        # compute parameters
        params = sum([p.numel() for p in self.parameters()])

        output = torch.zeros(
            size=(b, self.out_channels, out_h, out_w),
            dtype=input.dtype,
            device=input.device,
        )
        # print(macs)
        return output, params, macs


class Identity(BaseLayer):
    """
    This is a place-holder and returns the same tensor.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x

    def profile_module(self, x: Tensor) -> Tuple[Tensor, float, float]:
        return x, 0.0, 0.0


class LinearLayer(BaseLayer):
    """
    Applies a linear transformation to the input data
    Args:
        in_features (int): number of features in the input tensor
        out_features (int): number of features in the output tensor
        bias  (Optional[bool]): use bias or not
        channel_first (Optional[bool]): Channels are first or last dimension. If first, then use Conv2d
    Shape:
        - Input: :math:`(N, *, C_{in})` if not channel_first else :math:`(N, C_{in}, *)` where :math:`*` means any number of dimensions.
        - Output: :math:`(N, *, C_{out})` if not channel_first else :math:`(N, C_{out}, *)`
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: Optional[bool] = True,
            channel_first: Optional[bool] = False,
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

        self.in_features = in_features
        self.out_features = out_features
        self.channel_first = channel_first

        self.reset_params()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model.layer.linear-init",
            type=str,
            default="xavier_uniform",
            help="Init type for linear layers",
        )
        parser.add_argument(
            "--model.layer.linear-init-std-dev",
            type=float,
            default=0.01,
            help="Std deviation for Linear layers",
        )
        return parser

    def reset_params(self):
        if self.weight is not None:
            torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor):
        if self.channel_first:
            if not self.training:
                print("Channel-first mode is only supported during inference")
            if x.dim() != 4:
                print("Input should be 4D, i.e., (B, C, H, W) format")
            # only run during conversion
            with torch.no_grad():
                return F.conv2d(
                    input=x,
                    weight=self.weight.clone().detach().reshape(self.out_features, self.in_features, 1, 1),
                    bias=self.bias)
        else:
            x = F.linear(x, weight=self.weight, bias=self.bias)
        return x

    def __repr__(self):
        repr_str = (
            "{}(in_features={}, out_features={}, bias={}, channel_first={})".format(
                self.__class__.__name__,
                self.in_features,
                self.out_features,
                True if self.bias is not None else False,
                self.channel_first,
            )
        )
        return repr_str

    def profile_module(
            self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        out_size = list(input.shape)
        out_size[-1] = self.out_features
        params = sum([p.numel() for p in self.parameters()])
        macs = params
        output = torch.zeros(size=out_size, dtype=input.dtype, device=input.device)
        return output, params, macs


class GlobalPool(BaseLayer):
    """
    This layers applies global pooling over a 4D or 5D input tensor
    Args:
        pool_type (Optional[str]): Pooling type. It can be mean, rms, or abs. Default: `mean`
        keep_dim (Optional[bool]): Do not squeeze the dimensions of a tensor. Default: `False`
    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, 1, 1)` or :math:`(N, C, 1, 1, 1)` if keep_dim else :math:`(N, C)`
    """

    pool_types = ["mean", "rms", "abs"]

    def __init__(
            self,
            pool_type: Optional[str] = "mean",
            keep_dim: Optional[bool] = False,
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        if pool_type not in self.pool_types:
            print(
                "Supported pool types are: {}. Got {}".format(
                    self.pool_types, pool_type
                )
            )
        self.pool_type = pool_type
        self.keep_dim = keep_dim

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        cls_name = "{} arguments".format(cls.__name__)
        group = parser.add_argument_group(title=cls_name, description=cls_name)
        group.add_argument(
            "--model.layer.global-pool",
            type=str,
            default="mean",
            help="Which global pooling?",
        )
        return parser

    def _global_pool(self, x: Tensor, dims: List):
        if self.pool_type == "rms":  # root mean square
            x = x ** 2
            x = torch.mean(x, dim=dims, keepdim=self.keep_dim)
            x = x ** -0.5
        elif self.pool_type == "abs":  # absolute
            x = torch.mean(torch.abs(x), dim=dims, keepdim=self.keep_dim)
        else:
            # default is mean
            # same as AdaptiveAvgPool
            x = torch.mean(x, dim=dims, keepdim=self.keep_dim)
        return x

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            dims = [-2, -1]
        elif x.dim() == 5:
            dims = [-3, -2, -1]
        else:
            raise NotImplementedError("Currently 2D and 3D global pooling supported")
        return self._global_pool(x, dims=dims)

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        input = self.forward(input)
        return input, 0.0, 0.0

    def __repr__(self):
        return "{}(type={})".format(self.__class__.__name__, self.pool_type)


class LinearSelfAttention(BaseLayer):
    """
    This layer applies a self-attention with linear complexity, as described in `this paper <>`_
    This layer can be used for self- as well as cross-attention.
    Args:
        opts: command line arguments
        embed_dim (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        attn_dropout (Optional[float]): Dropout value for context scores. Default: 0.0
        bias (Optional[bool]): Use bias in learnable layers. Default: True
    Shape:
        - Input: :math:`(N, C, P, N)` where :math:`N` is the batch size, :math:`C` is the input channels,
        :math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
        - Output: same as the input
    .. note::
        For MobileViTv2, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
        in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
        we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
        expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
        channel-first to channel-last format in case of a linear layer.
    """

    def __init__(
            self,
            opts,
            embed_dim: int,
            attn_dropout: Optional[float] = 0.0,
            bias: Optional[bool] = True,
            *args,
            **kwargs
    ) -> None:
        super().__init__()

        self.qkv_proj = ConvLayer(
            opts=opts,
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )

        self.attn_dropout = Dropout(p=attn_dropout)
        self.out_proj = ConvLayer(
            opts=opts,
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )
        self.embed_dim = embed_dim

    def __repr__(self):
        return "{}(embed_dim={}, attn_dropout={})".format(
            self.__class__.__name__, self.embed_dim, self.attn_dropout.p
        )

    @staticmethod
    def visualize_context_scores(context_scores):
        # [B, 1, P, N]
        batch_size, channels, num_pixels, num_patches = context_scores.shape

        assert batch_size == 1, "For visualization purposes, use batch size of 1"
        assert (
                channels == 1
        ), "The inner-product between input and latent node (query) is a scalar"

        up_scale_factor = int(num_pixels ** 0.5)
        patch_h = patch_w = int(context_scores.shape[-1] ** 0.5)
        # [1, 1, P, N] --> [1, P, h, w]
        context_scores = context_scores.reshape(1, num_pixels, patch_h, patch_w)
        # Fold context scores [1, P, h, w] using pixel shuffle to obtain [1, 1, H, W]
        context_map = F.pixel_shuffle(context_scores, upscale_factor=up_scale_factor)
        # [1, 1, H, W] --> [H, W]
        context_map = context_map.squeeze()

        # For ease of visualization, we do min-max normalization
        min_val = torch.min(context_map)
        max_val = torch.max(context_map)
        context_map = (context_map - min_val) / (max_val - min_val)

        try:
            import cv2
            from glob import glob
            import os

            # convert from float to byte
            context_map = (context_map * 255).byte().cpu().numpy()
            context_map = cv2.resize(
                context_map, (80, 80), interpolation=cv2.INTER_NEAREST
            )

            colored_context_map = cv2.applyColorMap(context_map, cv2.COLORMAP_JET)
            # Lazy way to dump feature maps in attn_res folder. Make sure that directory is empty and copy
            # context maps before running on different image. Otherwise, attention maps will be overridden.
            res_dir_name = "attn_res"
            if not os.path.isdir(res_dir_name):
                os.makedirs(res_dir_name)
            f_name = "{}/h_{}_w_{}_index_".format(res_dir_name, patch_h, patch_w)

            files_cmap = glob(
                "{}/h_{}_w_{}_index_*.png".format(res_dir_name, patch_h, patch_w)
            )
            idx = len(files_cmap)
            f_name += str(idx)

            cv2.imwrite("{}.png".format(f_name), colored_context_map)
            return colored_context_map
        except ModuleNotFoundError as mnfe:
            print("Please install OpenCV to visualize context maps")
            return context_map

    def _forward_self_attn(self, x: Tensor, *args, **kwargs) -> Tensor:
        # [B, C, P, N] --> [B, h + 2d, P, N]
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        # Uncomment below line to visualize context scores
        # self.visualize_context_scores(context_scores=context_scores)
        context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]
        context_vector = key * context_scores
        # [B, d, P, N] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

    def _forward_cross_attn(
            self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        # x --> [B, C, P, N]
        # x_prev = [B, C, P, M]

        batch_size, in_dim, kv_patch_area, kv_num_patches = x.shape

        q_patch_area, q_num_patches = x.shape[-2:]

        assert (
                kv_patch_area == q_patch_area
        ), "The number of pixels in a patch for query and key_value should be the same"

        # compute query, key, and value
        # [B, C, P, M] --> [B, 1 + d, P, M]
        qk = F.conv2d(
            x_prev,
            weight=self.qkv_proj.block.conv.weight[: self.embed_dim + 1, ...],
            bias=self.qkv_proj.block.conv.bias[: self.embed_dim + 1, ...],
        )
        # [B, 1 + d, P, M] --> [B, 1, P, M], [B, d, P, M]
        query, key = torch.split(qk, split_size_or_sections=[1, self.embed_dim], dim=1)
        # [B, C, P, N] --> [B, d, P, N]
        value = F.conv2d(
            x,
            weight=self.qkv_proj.block.conv.weight[self.embed_dim + 1:, ...],
            bias=self.qkv_proj.block.conv.bias[self.embed_dim + 1:, ...],
        )

        # apply softmax along M dimension
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        # compute context vector
        # [B, d, P, M] * [B, 1, P, M] -> [B, d, P, M]
        context_vector = key * context_scores
        # [B, d, P, M] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

    def forward(
            self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        if x_prev is None:
            return self._forward_self_attn(x, *args, **kwargs)
        else:
            return self._forward_cross_attn(x, x_prev=x_prev, *args, **kwargs)

    def profile_module(self, input) -> Tuple[Tensor, float, float]:
        params = macs = 0.0

        qkv, p, m = module_profile(module=self.qkv_proj, x=input)
        params += p
        macs += m

        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        if self.out_proj is not None:
            out_p, p, m = module_profile(module=self.out_proj, x=value)
            params += p
            macs += m

        return input, params, macs


class Dropout(nn.Dropout):
    """
    This layer, during training, randomly zeroes some of the elements of the input tensor with probability `p`
    using samples from a Bernoulli distribution.
    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(N, *)` where :math:`N` is the batch size
        - Output: same as the input
    """

    def __init__(
            self, p: Optional[float] = 0.5, inplace: Optional[bool] = False, *args, **kwargs
    ) -> None:
        super().__init__(p=p, inplace=inplace)

    def profile_module(
            self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0


class Dropout2d(nn.Dropout2d):
    """
    This layer, during training, randomly zeroes some of the elements of the 4D input tensor with probability `p`
    using samples from a Bernoulli distribution.
    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(N, C, H, W)` where :math:`N` is the batch size, :math:`C` is the input channels,
            :math:`H` is the input tensor height, and :math:`W` is the input tensor width
        - Output: same as the input
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__(p=p, inplace=inplace)

    def profile_module(self, input: Tensor, *args, **kwargs) -> (Tensor, float, float):
        return input, 0.0, 0.0
