"""Tests for Brawn policies."""
import os

import dill
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace

DEFAULT_CHECKPOINT_PATH = os.path.expanduser(
    "~/brawn_artifacts/checkpoints/diffusion_policy_pick_sugar_2024_10_19.ckpt")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def load_workspace(checkpoint_path: str) -> TrainDiffusionUnetImageWorkspace:
    """Load workspace from checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location='cuda', pickle_module=dill)
    workspace = TrainDiffusionUnetImageWorkspace(state_dict['cfg'])
    workspace.load_payload(state_dict, exclude_keys=None, include_keys=None)
    return workspace


def get_policy_from_workspace(workspace: TrainDiffusionUnetImageWorkspace) -> BaseImagePolicy:
    """Load policy from workspace."""
    if workspace.cfg.training.use_ema:
        policy = workspace.ema_model
    else:
        policy = workspace.model

    policy.eval().to(DEVICE)
    policy.num_inference_steps = 16  # DDIM inference iterations
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    return policy


def generate_obs_dict(rng: np.random.RandomState, n_steps: int) -> dict:
    """Generate random observation dictionary."""
    obs_dict_np = {
        'image': rng.random_sample(size=(n_steps, 3, 240, 240)),
        'state': rng.random_sample(size=(n_steps, 7))
    }
    return dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(DEVICE))


def test_pick_sugar_runs(checkpoint_path: str = DEFAULT_CHECKPOINT_PATH):
    """[smoke]Test that the pick sugar policy runs."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")

    workspace: TrainDiffusionUnetImageWorkspace = load_workspace(checkpoint_path)
    policy: BaseImagePolicy = get_policy_from_workspace(workspace)

    rng = np.random.RandomState(0)
    obs_dict = generate_obs_dict(rng=rng, n_steps=policy.n_obs_steps)

    with torch.no_grad():
        result = policy.predict_action(obs_dict)
    action = result['action'][0].detach().to('cpu').numpy()
    print(action)


def test_pick_sugar_on_dataset(checkpoint_path: str = DEFAULT_CHECKPOINT_PATH):
    """Test that the pick sugar policy performs as expected on a training batch."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")

    workspace: TrainDiffusionUnetImageWorkspace = load_workspace(checkpoint_path)
    policy: BaseImagePolicy = get_policy_from_workspace(workspace)

    dataset = hydra.utils.instantiate(workspace.cfg.task.dataset)
    dataloader = DataLoader(dataset, **workspace.cfg.dataloader)  # training set

    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)  # needed?
    policy.eval().to(DEVICE)

    batch = next(iter(dataloader))
    batch = dict_apply(batch, lambda x: x.to(DEVICE, non_blocking=True))
    with torch.no_grad():
        loss = policy.compute_loss(batch).cpu().numpy()

    print(f'Loss: {loss}')
    assert loss < 0.01


if __name__ == '__main__':
    test_pick_sugar_runs()
    test_pick_sugar_on_dataset()
