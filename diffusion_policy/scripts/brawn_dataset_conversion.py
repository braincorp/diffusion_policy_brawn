"""Script for converting from the RLDS brawn dataset format into the diffusion policy format."""
import os
from typing import TypedDict

import click
import numpy as np
from tqdm import tqdm

import scipy.spatial.transform as st
from diffusion_policy.common.replay_buffer import ReplayBuffer

try:
    import tensorflow_datasets as tfds
except ImportError:
    raise ImportError(
        "Please install the tensorflow-datasets package to use this script.\n"
        ">> pip install tensorflow==2.17.0 tensorflow-datasets==4.9.3"
    )


class EpisodeDataDict(TypedDict):
    """Format of the episode data."""
    actions: np.ndarray  # Absolute actions N x 7 [translation (x, y, z), rotation (as rotvec), gripper (open/close)]
    states: np.ndarray  # States N x 7  [translation (x, y, z), rotation (as rotvec), gripper (position)]
    images: np.ndarray  # Images N x H x W x C
    instructions: np.ndarray  # Instructions N x 1


@click.command()
@click.option('-i', '--input_rlds_path', required=True, help='path to the RLDS dataset')
@click.option('-o', '--output_directory', required=True, help='directory in which the converted dataset will be stored')
def main(
        input_rlds_path: str,
        output_directory: str,
) -> None:
    """Convert the RLDS dataset into the diffusion policy format."""
    if not os.path.exists(input_rlds_path):
        raise FileNotFoundError(f"RLDS dataset path {input_rlds_path} does not exist.")

    if not os.path.exists(output_directory):
        raise FileNotFoundError(f"Output directory {output_directory} does not exist.")

    # Load the RLDS dataset
    rlds_dataset_name = os.path.basename(input_rlds_path)
    rlds_parent_directory = os.path.dirname(input_rlds_path)
    dataset_tfds = tfds.load(
        name=rlds_dataset_name,
        data_dir=rlds_parent_directory,
        with_info=False,
    )

    out_replay_buffer = ReplayBuffer.create_empty_numpy()
    for episode_index, episode in enumerate(tqdm(dataset_tfds['train'])):
        episode_metadata = tfds.as_numpy(episode['episode_metadata'])
        episode_actuation_type = episode_metadata['action_actuation_type'].decode('utf-8')
        if episode_actuation_type != 'position':
            print(f"Skipping episode {episode_index} because actuation type is not position.")
            continue

        episode_num_steps = len(episode['steps'])
        if episode_num_steps == 0:
            print(f"Skipping episode {episode_index} because it has no steps.")
            continue

        previous_action_absolute_translation = None
        previous_action_absolute_orientation = None

        episode_actions = []
        episode_states = []
        episode_images = []
        episode_instructions = []
        for step in episode['steps']:
            observation = step['observation']
            current_state_translation = observation['eef_translational_positions']
            current_state_orientation = st.Rotation.from_euler(
                seq='xyz',
                angles=observation['eef_rotational_positions']
            )
            current_state_gripper = observation['gripper_position']
            current_state = np.concatenate(
                [
                    current_state_translation,
                    current_state_orientation.as_rotvec(),
                    current_state_gripper[None]
                ]
            )

            # Sanity check that the current state matches the expected state
            if previous_action_absolute_translation is not None:
                if not np.allclose(current_state_translation, previous_action_absolute_translation):
                    raise ValueError(f"Current state translation does not match expected translation!")

                if not np.allclose(
                    current_state_orientation.as_matrix(),
                    previous_action_absolute_orientation.as_matrix(),
                    atol=1e-4
                ):
                    raise ValueError(f"Current state orientation does not match expected orientation!")

            action_vector = step['action']['action_vector']
            action_relative_translation = action_vector[:3]
            action_relative_orientation = st.Rotation.from_euler(
                seq='xyz',
                angles=action_vector[3:6]
            )
            action_gripper = action_vector[-1]

            action_absolute_translation = action_relative_translation + current_state_translation
            action_absolute_orientation = action_relative_orientation * current_state_orientation
            action_absolute = np.concatenate(
                [
                    action_absolute_translation,
                    action_absolute_orientation.as_rotvec(),
                    action_gripper[None]
                ]
            )

            episode_states.append(current_state)
            episode_actions.append(action_absolute)
            episode_images.append(observation['static_rgb_image'])
            episode_instructions.append(step['language_instruction'].numpy().decode())

            previous_action_absolute_translation = action_absolute_translation
            previous_action_absolute_orientation = action_absolute_orientation

        episode_data = EpisodeDataDict(
            actions=np.array(episode_actions),
            states=np.array(episode_states),
            images=np.array(episode_images),
            instructions=np.array(episode_instructions),
        )
        out_replay_buffer.add_episode(episode_data)

    output_zarr_path = os.path.join(output_directory, f"{rlds_dataset_name}.zarr.zip")
    out_replay_buffer.save_to_path(zarr_path=output_zarr_path, chunk_length=-1)


if __name__ == '__main__':
    main()
