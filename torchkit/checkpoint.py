import logging
import os
import signal
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Union

import torch

from .experiment import unique_id


def get_files(
    d: Path,
    pattern: str,
    sort_lexicographical: bool = False,
    sort_numerical: bool = False,
) -> List[Path]:
    """Return a list of files in a given directory.

    Args:
        d: The path to the directory.
        pattern: The wildcard to filter files with.
        sort_lexicographical: Lexicographical sort.
        sort_numerical: Numerical sort.
    """
    pass


class Checkpoint:
    """Save and restore PyTorch objects implementing a `state_dict` method."""

    def __init__(self, **kwargs) -> None:
        """Constructor.

        Accepts keyword arguments whose values are objects that contain a
        `state_dict` attribute and thus can be serialized to disk.

        Args:
            kwargs: Keyword arguments are set as attributes of this object,
                and are saved with the checkpoint. Values must have a
                `state_dict` attribute.

        Raises:
            ValueError: If objects in `kwargs` do not have a `state_dict`
                attribute.
        """
        for k, v in sorted(kwargs.items()):
            if not getattr(v, "state_dict"):
                raise ValueError(f"{k} does not have a state_dict attribute.")
            setattr(self, k, v)

    def save(self, save_path: Path) -> None:
        """Save a state to disk.

        Modified from brentyi/fannypack.

        Args:
            save_path: The name of the checkpoint to save.
        """
        pass

    def restore(self, save_path: Union[str, Path]) -> bool:
        """Restore a state from a saved checkpoint.

        Args:
            save_path: The filepath to the saved checkpoint.

        Returns:
            True if restoring was successful or partially (not all
            checkpointables could be restored) successful and False otherwise.
        """
        pass


# TODO(kevin): Add saving of best checkpoint based on specified metric.
class CheckpointManager:
    """
    Periodically save PyTorch checkpointables (any object that implements a
    `state_dict` method) to disk and restore them to resume training.

    Note: This is a re-implementation of `2`_.

    Example usage::

        from torchkit.checkpoint import CheckpointManager

        # Create a checkpoint manager instance.
        checkpoint_manager = checkpoint.CheckpointManager(
            checkpoint_dir,
            device,
            model=model,
            optimizer=optimizer,
        )

        # Restore last checkpoint if it exists.
        global_step = checkpoint_manager.restore_or_initialize()
        for global_step in range(1000):
            # forward pass + loss computation

            # Save a checkpoint every N iters.
            if not global_step % N:
                checkpoint_manager.save(global_step)

    .. _2: https://www.tensorflow.org/api_docs/python/tf/train/CheckpointManager/
    """

    def __init__(
        self,
        directory: str,
        max_to_keep: int = 10,
        **checkpointables: Any,
    ) -> None:
        """Constructor.

        Args:
            directory: The directory in which checkpoints will be saved.
            max_to_keep: The maximum number of checkpoints to keep.
                Amongst all saved checkpoints, checkpoints will be deleted
                oldest first, until `max_to_keep` remain.
            checkpointables: Keyword args with checkpointable PyTorch objects.
        """
        assert max_to_keep > 0, "max_to_keep should be a positive integer."

        self.directory = Path(directory).absolute()
        self.max_to_keep = max_to_keep
        self.checkpoint = Checkpoint(**checkpointables)

        # Create checkpoint directory if it doesn't already exist.
        self.directory.mkdir(parents=True, exist_ok=True)

    def restore_or_initialize(self) -> int:
        """Restore items in checkpoint from the latest checkpoint file.

        Returns:
            The global iteration step. This is parsed from the latest checkpoint
            file if one is found, else 0 is returned.
        """
        pass

    def save(self, global_step: int) -> None:
        """Create a new checkpoint.

        Args:
            global_step: The iteration number which will be used to name the
                checkpoint.
        """
        pass

    def _trim_checkpoints(self) -> None:
        """Trim older checkpoints until `max_to_keep` remain."""
        pass

    def load_latest_checkpoint(self) -> None:
        """Load the last saved checkpoint."""
        pass

    def load_checkpoint_at(self, global_step: int) -> None:
        """Load a checkpoint at a given global step."""
        pass

    @property
    def latest_checkpoint(self) -> Optional[Path]:
        """Get the last saved checkpoint."""
        pass

    @staticmethod
    def list_checkpoints(directory: Union[Path, str]) -> List[Path]:
        """List all checkpoints in a checkpoint directory."""
        pass
