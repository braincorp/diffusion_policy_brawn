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


class BrawnCarrotPlateImageDataset(BaseImageDataset):
    def __init__(
            self,
            shape_meta: dict,
            dataset_path: str,
            horizon: int = 1,
            pad_before: int = 0,
            pad_after: int = 0,
            n_obs_steps: Optional[int] = None,
            n_latency_steps: int = 0,
            use_cache: bool = False,
            seed: int = 42,
            val_ratio: float = 0.0,
            max_train_episodes: Optional[int] = None,
    ):
        if not os.path.isdir(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist")

        replay_buffer: ReplayBuffer
        if use_cache:
            # fingerprint shape_meta
            shape_meta_json = json.dumps(OmegaConf.to_container(shape_meta), sort_keys=True)
            shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
            cache_zarr_path = os.path.join(dataset_path, shape_meta_hash + '.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        replay_buffer = _load_replay_buffer(
                            dataset_path=dataset_path,
                            shape_meta=shape_meta,
                            store=zarr.MemoryStore()
                        )
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _load_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=zarr.MemoryStore()
            )




def _load_replay_buffer(dataset_path: str, shape_meta: dict, store: Optional[zarr.ABSStore]) -> ReplayBuffer:
    """Load replay buffer from dataset path."""
    # parse shape meta
    rgb_keys = []
    lowdim_keys = []
    out_resolutions = {}
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type_ = attr.get('type', 'low_dim')
        shape = tuple(attr.get('shape'))
        if type_ == 'rgb':
            rgb_keys.append(key)
            c, h, w = shape
            out_resolutions[key] = (w, h)
        elif type_ == 'low_dim':
            lowdim_keys.append(key)
            if 'pose' in key:
                assert tuple(shape) in [(2,), (6,)]

    action_shape = tuple(shape_meta['action']['shape'])
    assert action_shape in [(2,), (6,)]

    # load data
    cv2.setNumThreads(1)
    with threadpool_limits(1):
        replay_buffer = real_data_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,
            out_resolutions=out_resolutions,
            lowdim_keys=lowdim_keys + ['action'],
            image_keys=rgb_keys
        )

    return replay_buffer
