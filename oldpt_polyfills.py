import torch

from torch.nn.modules.module import Module
from torch.nn import functional as F
from torch import Tensor

#https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
class SiLU(Module):
    r"""Applies the silu function, element-wise.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.SiLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(SiLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.silu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
    
torch.nn.modules.activation.SiLU = SiLU
torch.nn.SiLU = SiLU


#https://github.com/aakibinesar/Sigmoid-Linear-Unit/blob/main/MNIST_DNN.py#L77
def silu(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Applies the silu function, element-wise.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    See :class:`~torch.nn.SiLU` for more details.
    
    original pytorch code is referring to C code, so
    this was taken from here https://github.com/aakibinesar/Sigmoid-Linear-Unit/blob/main/MNIST_DNN.py#L77"""
    return input * (1 / (1 + (torch.exp(-input))))

torch.nn.functional.silu = silu


#https://pytorch.org/docs/stable/generated/torch.nn.Hardswish.html
class Hardswish(Module):
    r"""Applies the hardswish function, element-wise, as described in the paper:

    `Searching for MobileNetV3`_.

    .. math::
        \text{Hardswish}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            x & \text{if~} x \ge +3, \\
            x \cdot (x + 3) /6 & \text{otherwise}
        \end{cases}

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.Hardswish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    .. _`Searching for MobileNetV3`:
        https://arxiv.org/abs/1905.02244
    """
    __constants__ = ['inplace']

    inplace: bool

    def __init__(self, inplace : bool = False) -> None:
        super(Hardswish, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.hardswish(input, self.inplace)

torch.nn.Hardswish = Hardswish

#the code bellow is necessary only for training
#it is intended to replace missing function and might be not accurate

torch.cuda.memory_reserved = torch.cuda.memory_allocated