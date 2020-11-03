from typing import Dict

from numpy.random.mtrand import RandomState

from robogym.envs.rearrange.datasets.objects.base import ObjectDataset
from robogym.utils.env_utils import get_function


def get_envstate_datasets(
    dataset_config: Dict[str, dict],
    object_datasets: Dict[str, ObjectDataset],
    random_state: RandomState,
):
    datasets = {}
    for key in dataset_config:
        datasets[key] = get_function(dataset_config[key])(
            object_datasets=object_datasets, random_state=random_state
        )
    return datasets
