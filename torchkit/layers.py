from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

ConvType = Union[torch.nn.modules.conv.Conv2d, torch.nn.modules.conv.Conv3d]
Tensor = torch.Tensor


def _conv(
    dim: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    dilation: int = 1,
    bias: bool = True,
) -> ConvType:
    """`same` convolution, i.e. output shape equals input shape.

    Args:
        dim: The dimension of the convolution: 2 is conv2d, 3 is conv3d.
        in_planes: The number of input feature maps.
        out_planes: The number of output feature maps.
        kernel_size: The filter size.
        stride: The filter stride.
        dilation: The filter dilation factor.
        bias: Whether to add a bias.
    """
    pass


def conv2d(*args, **kwargs) -> torch.nn.modules.conv.Conv2d:
    """`same` 2D convolution, i.e. output shape equals input shape.

    Args:
        in_planes: The number of input feature maps.
        out_planes: The number of output feature maps.
        kernel_size: The filter size.
        stride: The filter stride.
        dilation: The filter dilation factor.
        bias: Whether to add a bias.
    """
    pass


def conv3d(*args, **kwargs) -> torch.nn.modules.conv.Conv3d:
    """`same` 3D convolution, i.e. output shape equals input shape.

    Args:
        in_planes: The number of input feature maps.
        out_planes: The number of output feature maps.
        kernel_size: The filter size.
        stride: The filter stride.
        dilation: The filter dilation factor.
        bias: Whether to add a bias.
    """
    pass


class Flatten(nn.Module):
    """Flattens convolutional feature maps for fully-connected layers.

    This is a convenience module meant to be plugged into a
    `torch.nn.Sequential` model.

    Example usage::

        import torch.nn as nn
        from torchkit import layers

        # Assume an input of shape (3, 28, 28).
        net = nn.Sequential(
            layers.conv2d(3, 8, kernel_size=3),
            nn.ReLU(),
            layers.conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            layers.Flatten(),
            nn.Linear(28*28*16, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass


class SpatialSoftArgmax(nn.Module):
    """Spatial softmax as defined in `1`_.

    Concretely, the spatial softmax of each feature map is used to compute a
    weighted mean of the pixel locations, effectively performing a soft arg-max
    over the feature dimension.

    .. _1: https://arxiv.org/abs/1504.00702
    """

    def __init__(self, normalize: bool = False):
        """Constructor.

        Args:
            normalize: Whether to use normalized image coordinates, i.e.
                coordinates in the range `[-1, 1]`.
        """
        super().__init__()

        self.normalize = normalize

    def _coord_grid(
        self,
        h: int,
        w: int,
        device: torch.device,
    ) -> Tensor:
        pass

    def forward(self, x: Tensor) -> Tensor:
        pass


class _GlobalMaxPool(nn.Module):
    """Global max pooling layer."""

    def __init__(self, dim):
        super().__init__()

        if dim == 1:
            self._pool = F.max_pool1d
        elif dim == 2:
            self._pool = F.max_pool2d
        elif dim == 3:
            self._pool = F.max_pool3d
        else:
            raise ValueError("{}D is not supported.")

    def forward(self, x: Tensor) -> Tensor:
        pass


class GlobalMaxPool1d(_GlobalMaxPool):
    """Global max pooling operation for temporal or 1D data."""

    def __init__(self):
        super().__init__(dim=1)


class GlobalMaxPool2d(_GlobalMaxPool):
    """Global max pooling operation for spatial or 2D data."""

    def __init__(self):
        super().__init__(dim=2)


class GlobalMaxPool3d(_GlobalMaxPool):
    """Global max pooling operation for 3D data."""

    def __init__(self):
        super().__init__(dim=3)


class _GlobalAvgPool(nn.Module):
    """Global average pooling layer."""

    def __init__(self, dim):
        super().__init__()

        if dim == 1:
            self._pool = F.avg_pool1d
        elif dim == 2:
            self._pool = F.avg_pool2d
        elif dim == 3:
            self._pool = F.avg_pool3d
        else:
            raise ValueError("{}D is not supported.")

    def forward(self, x: Tensor) -> Tensor:
        pass


class GlobalAvgPool1d(_GlobalAvgPool):
    """Global average pooling operation for temporal or 1D data."""

    def __init__(self):
        super().__init__(dim=1)


class GlobalAvgPool2d(_GlobalAvgPool):
    """Global average pooling operation for spatial or 2D data."""

    def __init__(self):
        super().__init__(dim=2)


class GlobalAvgPool3d(_GlobalAvgPool):
    """Global average pooling operation for 3D data."""

    def __init__(self):
        super().__init__(dim=3)


class CausalConv1d(nn.Conv1d):
    """A causal a.k.a. masked 1D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        """Constructor.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            kernel_size: The filter size.
            stride: The filter stride.
            dilation: The filter dilation factor.
            bias: Whether to add the bias term or not.
        """
        self.__padding = (kernel_size - 1) * dilation

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        pass
