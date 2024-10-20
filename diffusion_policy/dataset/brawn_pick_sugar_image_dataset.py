from __future__ import annotations

import copy
from typing import Dict, Optional

import numpy as np
import torch

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class BrawnPickSugarImageDataset(BaseImageDataset):
    def __init__(
            self,
            zarr_path: str, # path to zarr file
            horizon: int = 1,
            pad_before: int = 0,
            pad_after: int = 0,
            seed: int = 42,
            validation_ratio: float = 0.0,
            max_train_episodes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['actions', 'states', 'images'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=validation_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            data=self.replay_buffer['actions'],
            mode=mode,
        )
        normalizer['image'] = get_image_range_normalizer()
        normalizer['state'] = SingleFieldLinearNormalizer.create_fit(
            data=self.replay_buffer['states'],
            mode=mode
        )
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs:
                key: T, *
            action: T, Da
        """
        sample = self.sampler.sample_sequence(idx)
        data = {
            'obs': {
                'image': np.moveaxis(sample['images'], -1, 1) / 255,
                'state': sample['states'].astype(np.float32)
            },
            'action': sample['actions']
        }
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def _test_dataset():
    """Sanity check that the dataset is working"""
    import os
    from matplotlib import pyplot as plt

    zarr_path = os.path.expanduser('~/brawn_artifacts/datasets/dobot_nova5/episodes_pick_bottled_sugar_lab_above/episodes_pick_bottled_sugar_lab_above_one_episode_per_manilog_openvla_rlds.zarr.zip')
    dataset = BrawnPickSugarImageDataset(zarr_path=zarr_path, horizon=16)
    print(f"Number of episodes: {len(dataset)}")

    normalizer = dataset.get_normalizer()
    nactions = normalizer['action'].normalize(dataset.replay_buffer['actions'])
    for dim in range(nactions.shape[-1]):
        plt.plot(nactions[:,dim], label=f"dim {dim}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    _test_dataset()
