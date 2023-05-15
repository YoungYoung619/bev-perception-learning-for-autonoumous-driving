from torch import nn, Tensor
from typing import Optional, Union, Tuple
from torch.nn import functional as F
from torch import nn
from typing import Optional

class GELU(nn.GELU):
    """
    Applies the `Gaussian Error Linear Units <https://arxiv.org/abs/1606.08415>`_ function
    """

    def __init__(self) -> None:
        super().__init__()

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0


class Hardsigmoid(nn.Hardsigmoid):
    """
    Applies the `Hard Sigmoid <https://arxiv.org/abs/1511.00363v3>`_ function
    """

    def __init__(self, inplace: Optional[bool] = False) -> None:
        super().__init__(inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        if hasattr(F, "hardsigmoid"):
            return F.hardsigmoid(input, self.inplace)
        else:
            return F.relu(input + 3) / 6

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0


class Hardswish(nn.Hardswish):
    """
    Applies the HardSwish function, as described in the paper
    `Searching for MobileNetv3 <https://arxiv.org/abs/1905.02244>`_
    """

    def __init__(self, inplace: Optional[bool] = False) -> None:
        super().__init__(inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        if hasattr(F, "hardswish"):
            return F.hardswish(input, self.inplace)
        else:
            x_hard_sig = F.relu(input + 3) / 6
            return input * x_hard_sig

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0


class LeakyReLU(nn.LeakyReLU):
    """
    Applies a leaky relu function. See `Rectifier Nonlinearities Improve Neural Network Acoustic Models`
    for more details.
    """

    def __init__(
            self, negative_slope: Optional[float] = 1e-2, inplace: Optional[bool] = False
    ) -> None:
        super().__init__(negative_slope=negative_slope, inplace=inplace)

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0


class PReLU(nn.PReLU):
    """
    Applies the `Parametric Rectified Linear Unit <https://arxiv.org/abs/1502.01852>`_ function
    """

    def __init__(
            self, num_parameters: Optional[int] = 1, init: Optional[float] = 0.25
    ) -> None:
        super().__init__(num_parameters=num_parameters, init=init)

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0


class ReLU(nn.ReLU):
    """
    Applies Rectified Linear Unit function
    """

    def __init__(self, inplace: Optional[bool] = False) -> None:
        super().__init__(inplace=inplace)

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0


class ReLU6(nn.ReLU6):
    """
    Applies the ReLU6 function
    """

    def __init__(self, inplace: Optional[bool] = False) -> None:
        super().__init__(inplace=inplace)

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0


class Sigmoid(nn.Sigmoid):
    """
    Applies the sigmoid function
    """

    def __init__(self):
        super().__init__()

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0


class Tanh(nn.Tanh):
    """
    Applies Tanh function
    """

    def __init__(self):
        super(Tanh, self).__init__()

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0


class Swish(nn.SiLU):
    """
    Applies the `Swish (also known as SiLU) <https://arxiv.org/abs/1702.03118>`_ function.
    """

    def __init__(self, inplace: Optional[bool] = False) -> None:
        super().__init__(inplace=inplace)

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0