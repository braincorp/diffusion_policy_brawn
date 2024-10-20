"""Script for converting from the RLDS brawn dataset format into the diffusion policy format."""
# TODO: Start here! Then implement brawn_carrot_plate_image_dataset.py
import os
from typing import TypedDict

import click
import numpy as np
from tqdm import tqdm

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
    actions_absolute: np.ndarray
    actions_relative: np.ndarray
    images: np.ndarray
    instructions: np.ndarray


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
    import pdb; pdb.set_trace()
    for episode_index, episode in enumerate(tqdm(dataset_tfds['train'])):
        episode_metadata = tfds.as_numpy(episode['episode_metadata'])
        episode_actuation_type = episode_metadata['action_actuation_type'].decode('utf-8')
        if episode_actuation_type != 'position':
            print(f"Skipping episode {episode_index} because actuation type is not position.")
            continue

        episode_actions_relative = []
        episode_actions_absolute = []
        episode_images = []
        episode_instructions = []
        for step in episode['steps']:
            observation = step['observation']
            action_relative = step['action']['action_vector']
            action_absolute = np.concatenate(
                [
                    observation['eef_translational_positions'],
                    observation['eef_rotational_positions'],
                    action_relative[-1, None]
                ]
            )

            episode_actions_relative.append(action_relative)
            episode_actions_absolute.append(action_absolute)
            episode_images.append(observation['static_rgb_image'])
            episode_instructions.append(step['language_instruction'].numpy().decode())

        episode_data = EpisodeDataDict(
            actions_absolute=np.array(episode_actions_absolute),
            actions_relative=np.array(episode_actions_relative),
            images=np.array(episode_images),
            instructions=np.array(episode_instructions),
        )
        out_replay_buffer.add_episode(episode_data)

    output_zarr_path = os.path.join(output_directory, f"{rlds_dataset_name}.zarr.zip")
    out_replay_buffer.save_to_path(zarr_path=output_zarr_path, chunk_length=-1)


if __name__ == '__main__':
    main()
