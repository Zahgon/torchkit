"""Tools for visualizing resnet feature maps and ViT attention weights."""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms as T


def _load_model(
    model_name: str,
    device: torch.device,
):
    pass


def visualize_attention(
    model_name: str,
    image: Union[str, np.ndarray],
    image_size: Tuple[int, int] = (480, 480),
    resnet_layer_idx: Optional[int] = -2,
) -> np.ndarray:
    # Load the model.
    pass
