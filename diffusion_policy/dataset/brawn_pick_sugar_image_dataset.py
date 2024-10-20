from __future__ import annotations
from typing import Dict, List, Optional
import torch
import numpy as np
import zarr
import os
import shutil

from evdev.ff import Replay
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
import cv2
import json
import hashlib
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)


# TODO: The big question - is it easier to convert rlds into replay buffers, or just use rlds in the background?
#  Probably easier to convert first, so then can use the same framework as the other datasets
class BrawnCarrotPlateImageDataset(BaseImageDataset):
    def get_validation_dataset(self) -> BaseImageDataset:
        return BaseImageDataset()  # return an empty dataset by default

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        # can compute dataset stats like in openvla
        raise NotImplementedError("TODO")

    def get_all_actions(self) -> torch.Tensor:
        # probably not needed
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError("TODO")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs:
                key: T, *
            action: T, Da
        """
        raise NotImplementedError("TODO")