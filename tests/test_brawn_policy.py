"""Tests for Brawn policies."""
import os

import dill
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace


DEFAULT_CHECKPOINT_PATH = os.path.expanduser("~/brawn_artifacts/checkpoints/checkpoints/diffusion_policy_pick_sugar_2024_10_19.ckpt")


def load_workspace(checkpoint_path: str) -> TrainDiffusionUnetImageWorkspace:
    """Load workspace from checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location='cuda',pickle_module=dill)
    workspace = TrainDiffusionUnetImageWorkspace(state_dict['cfg'])
    workspace.load_payload(state_dict, exclude_keys=None, include_keys=None)
    return workspace


def get_policy_from_workspace(workspace: TrainDiffusionUnetImageWorkspace) -> BaseImagePolicy:
    """Load policy from workspace."""
    if workspace.cfg.training.use_ema:
        policy = workspace.ema_model
    else:
        policy = workspace.model

    policy.eval().to(torch.device('cuda'))
    policy.num_inference_steps = 16  # DDIM inference iterations
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    return policy


def test_pick_sugar_runs(checkpoint_path: str = DEFAULT_CHECKPOINT_PATH):
    """[smoke]Test that the pick sugar policy runs."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")

    workspace: TrainDiffusionUnetImageWorkspace = load_workspace(checkpoint_path)
    policy: BaseImagePolicy = get_policy_from_workspace(workspace)

    rng = np.random.RandomState(0)
    image = rng.random_sample(size=(policy.n_obs_steps, 3, 240, 240))
    state = rng.random_sample(size=(policy.n_obs_steps, 7))
    obs_dict_np = {
        'image': image,
        'state': state,
    }

    obs_dict = dict_apply(obs_dict_np,
                          lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
    action = result['action'][0].detach().to('cpu').numpy()
    print(action)


if __name__ == '__main__':
    test_pick_sugar_runs()
