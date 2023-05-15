from torch import nn, Tensor, Size
from typing import Optional, Tuple, Union, List


class BatchNorm2d(nn.BatchNorm2d):
    """
    Applies a `Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over a 4D input tensor
    Args:
        num_features (Optional, int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``
    Shape:
        - Input: :math:`(N, C, H, W)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        :math:`H` is the input height, and :math:`W` is the input width
        - Output: same shape as the input
    """

    def __init__(
            self,
            num_features: int,
            eps: Optional[float] = 1e-5,
            momentum: Optional[float] = 0.1,
            affine: Optional[bool] = True,
            track_running_stats: Optional[bool] = True,
            *args,
            **kwargs
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        # Since normalization layers can be fused, we do not count their operations
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0


class BatchNorm1d(nn.BatchNorm1d):
    """
    Applies a `Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over a 2D or 3D input tensor
    Args:
        num_features (Optional, int): :math:`C` from an expected input of size :math:`(N, C)` or :math:`(N, C, L)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``
    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)` where :math:`N` is the batch size,
        :math:`C` is the number of input channels,  and :math:`L` is the sequence length
        - Output: same shape as the input
    """

    def __init__(
            self,
            num_features: int,
            eps: Optional[float] = 1e-5,
            momentum: Optional[float] = 0.1,
            affine: Optional[bool] = True,
            track_running_stats: Optional[bool] = True,
            *args,
            **kwargs
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        # Since normalization layers can be fused, we do not count their operations
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0


class BatchNorm3d(nn.BatchNorm3d):
    def __init__(
            self,
            num_features: int,
            eps: Optional[float] = 1e-5,
            momentum: Optional[float] = 0.1,
            affine: Optional[bool] = True,
            track_running_stats: Optional[bool] = True,
            *args,
            **kwargs
    ) -> None:
        """
        Applies a `Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over a 5D input tensor
        Args:
            num_features (Optional, int): :math:`C` from an expected input of size :math:`(N, C, D, H, W)`
            eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
            momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
            affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
            track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``
        Shape:
            - Input: :math:`(N, C, D, H, W)` where :math:`N` is the batch size, :math:`C` is the number of input
            channels, :math:`D` is the input depth, :math:`H` is the input height, and :math:`W` is the input width
            - Output: same shape as the input
        """
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        # Since normalization layers can be fused, we do not count their operations
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0


class InstanceNorm2d(nn.InstanceNorm2d):
    """
    Applies a `Instance Normalization <https://arxiv.org/abs/1607.08022>`_ over a 4D input tensor
    Args:
        num_features (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``
    Shape:
        - Input: :math:`(N, C, H, W)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        :math:`H` is the input height, and :math:`W` is the input width
        - Output: same shape as the input
    """

    def __init__(
            self,
            num_features: int,
            eps: Optional[float] = 1e-5,
            momentum: Optional[float] = 0.1,
            affine: Optional[bool] = True,
            track_running_stats: Optional[bool] = True,
            *args,
            **kwargs
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        # Since normalization layers can be fused, we do not count their operations
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0


class InstanceNorm1d(nn.InstanceNorm1d):
    """
    Applies a `Instance Normalization <https://arxiv.org/abs/1607.08022>`_ over a 2D or 3D input tensor
    Args:
        num_features (int): :math:`C` from an expected input of size :math:`(N, C)` or :math:`(N, C, L)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``
    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)` where :math:`N` is the batch size, :math:`C` is the number
        of input channels,  and :math:`L` is the sequence length
    - Output: same shape as the input
    """

    def __init__(
            self,
            num_features: int,
            eps: Optional[float] = 1e-5,
            momentum: Optional[float] = 0.1,
            affine: Optional[bool] = True,
            track_running_stats: Optional[bool] = True,
            *args,
            **kwargs
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        # Since normalization layers can be fused, we do not count their operations
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0


class LayerNorm(nn.LayerNorm):
    """
    Applies `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ over a input tensor
    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size
            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
    Shape:
        - Input: :math:`(N, *)` where :math:`N` is the batch size
        - Output: same shape as the input
    """

    def __init__(
            self,
            normalized_shape: Union[int, List[int], Size],
            eps: Optional[float] = 1e-5,
            elementwise_affine: Optional[bool] = True,
            *args,
            **kwargs
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0


class LayerNorm2D(nn.GroupNorm):
    """
    Applies `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ over a 4D input tensor
    Args:
        num_features (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
    Shape:
        - Input: :math:`(N, C, H, W)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        :math:`H` is the input height, and :math:`W` is the input width
        - Output: same shape as the input
    """

    def __init__(
            self,
            num_features: int,
            eps: Optional[float] = 1e-5,
            elementwise_affine: Optional[bool] = True,
            *args,
            **kwargs
    ) -> None:
        super().__init__(
            num_channels=num_features, eps=eps, affine=elementwise_affine, num_groups=1
        )
        self.num_channels = num_features

    def __repr__(self):
        return "{}(num_channels={}, eps={}, affine={})".format(
            self.__class__.__name__, self.num_channels, self.eps, self.affine
        )

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0


class SyncBatchNorm(nn.SyncBatchNorm):
    """
    Applies a `Syncronized Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over the input tensor
    Args:
        num_features (Optional, int): :math:`C` from an expected input of size :math:`(N, C, *)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``
    Shape:
        - Input: :math:`(N, C, *)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        :math:`*` is the remaining input dimensions
        - Output: same shape as the input
    """

    def __init__(
            self,
            num_features: int,
            eps: Optional[float] = 1e-5,
            momentum: Optional[float] = 0.1,
            affine: Optional[bool] = True,
            track_running_stats: Optional[bool] = True,
            *args,
            **kwargs
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        # Since normalization layers can be fused, we do not count their operations
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0


class GroupNorm(nn.GroupNorm):
    """
    Applies a `Group Normalization <https://arxiv.org/abs/1803.08494>`_ over an input tensor
    Args:
        num_groups (int): number of groups to separate the input channels into
        num_channels (int): :math:`C` from an expected input of size :math:`(N, C, *)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
    Shape:
        - Input: :math:`(N, C, *)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        and :math:`*` is the remaining dimensions of the input tensor
        - Output: same shape as the input
    .. note::
        GroupNorm is the same as LayerNorm when `num_groups=1` and it is the same as InstanceNorm when
        `num_groups=C`.
    """

    def __init__(
            self,
            num_groups: int,
            num_channels: int,
            eps: Optional[float] = 1e-5,
            affine: Optional[bool] = True,
            *args,
            **kwargs
    ) -> None:
        super().__init__(
            num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine
        )

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        # Since normalization layers can be fused, we do not count their operations
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0
