import os.path as osp
from typing import Type, Union, cast

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

Tensor = torch.Tensor
ImageType = Union[Tensor, np.ndarray]


class Logger:
    """A Tensorboard-based logger."""

    def __init__(self, log_dir: str, force_write: bool = False) -> None:
        """Constructor.

        Args:
            log_dir: The directory in which to store Tensorboard logs.
            force_write: Whether to force write to an already existing log dir.
                Set to `True` if resuming training.
        """
        # Setup the summary writer.
        if osp.exists(log_dir) and not force_write:
            raise ValueError(
                "You might be overwriting a directory that already "
                "has train_logs. Please provide a new experiment name "
                "or set --resume to True when launching train script."
            )
        self._writer = SummaryWriter(log_dir)

    def close(self) -> None:
        pass

    def flush(self) -> None:
        pass

    def log_scalar(
        self,
        scalar: Union[Tensor, float],
        global_step: int,
        name: str,
        prefix: str = "",
    ) -> None:
        """Log a scalar value.

        Args:
            scalar: A scalar `torch.Tensor` or float.
            global_step: The training iteration step.
            name: The name of the logged scalar.
            prefix: A prefix to prepend to the logged scalar.
        """
        pass

    def log_image(
        self,
        image: ImageType,
        global_step: int,
        name: str,
        prefix: str = "",
        nrow: int = 5,
    ) -> None:
        """Log an image or batch of images.

        Args:
            image: A numpy ndarray or a torch Tensor. If the image is 4D (i.e.
                batched), it will be converted to a 3D image using make_grid.
                The numpy array should be in channel-last format while the torch
                Tensor should be in channel-first format.
            global_step: The training iteration step.
            name: The name of the logged image(s).
            prefix: A prefix to prepend to the logged image(s).
            nrow: The number of images displayed in each row of the grid if the
                input image is 4D.
        """
        pass

    def log_video(
        self,
        video,
        global_step: int,
        name: str,
        prefix: str = "",
        fps: int = 4,
    ) -> None:
        """Log a sequence of images or a batch of sequence of images.

        Args:
            video: A torch Tensor or numpy ndarray. The numpy array should be in
                channel-last format while the torch Tensor should be in
                channel-first format. Should be either a single sequence of
                images of shape (T, CHW/HWC) or a batch of sequences of shape
                (B, T, CHW/HWC). The batch of sequences will get converted to
                one grid sequence of images.
            global_step: The training iteration step.
            name: The name of the logged video(s).
            prefix: A prefix to prepend to the logged video(s).
            fps: The frames per second.
        """
        pass

    def log_learning_rate(
        self,
        optimizer: Type[torch.optim.Optimizer],
        global_step: int,
        prefix: str = "",
    ) -> None:
        """Log the learning rate.

        Args:
            optimizer: An optimizer.
            global_step: The training iteration step.
        """
        pass
