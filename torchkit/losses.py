import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def one_hot(
    y: Tensor,
    K: int,
    smooth_eps: float = 0,
) -> Tensor:
    """One-hot encodes a tensor, with optional label smoothing.

    Args:
        y (Tensor): A tensor containing the ground-truth labels of shape `(N,)`, i.e.
            one label for each element in the batch.
        K (int): The number of classes.
        smooth_eps (float, optional): Label smoothing factor in `[0, 1]` range. Defaults
            to 0, which corresponds to no label smoothing.

    Returns:
        Tensor: The one-hot encoded tensor.
    """
    pass


def cross_entropy(
    logits: Tensor,
    labels: Tensor,
    smooth_eps: float = 0,
    reduction: str = "mean",
) -> Tensor:
    """Cross-entropy loss with support for label smoothing.

    Args:
        logits (Tensor): A `FloatTensor` containing the raw logits, i.e. no softmax has
            been applied to the model output. The tensor should be of shape
            `(N, K)` where K is the number of classes.
        labels (Tensor): A rank-1 `LongTensor` containing the ground truth labels.
        smooth_eps (float, optional): The label smoothing factor in `[0, 1]` range.
            Defaults to 0.
        reduction (str, optional): The reduction strategy on the final loss tensor.
            Defaults to "mean".

    Returns:
        If reduction is `none`, a 2D Tensor.
        If reduction is `sum`, a 1D Tensor.
        If reduction is `mean`, a scalar 1D Tensor.
    """
    pass


def huber_loss(
    input: Tensor,
    target: Tensor,
    delta: float,
    reduction: str = "mean",
) -> Tensor:
    """Huber loss with tunable margin, as defined in `1`_.

    Args:
        input (Tensor): A FloatTensor representing the model output.
        target (Tensor): A FloatTensor representing the target values.
        delta (float): Given the tensor difference `diff`, delta is the value at which
            we incur a quadratic penalty if `diff` is at least delta and a
            linear penalty otherwise.
        reduction (str, optional): The reduction strategy on the final loss tensor.
            Defaults to "mean".

    Returns:
        If reduction is `none`, a 2D Tensor.
        If reduction is `sum`, a 1D Tensor.
        If reduction is `mean`, a scalar 1D Tensor.

    .. _1: https://en.wikipedia.org/wiki/Huber_loss
    """
    pass
