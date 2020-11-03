from typing import Dict

from numpy.random.mtrand import RandomState

from robogym.utils.env_utils import get_function


def get_object_datasets(object_config: Dict[str, dict], random_state: RandomState):
    datasets = {}
    for key in object_config:
        datasets[key] = get_function(object_config[key])(random_state=random_state)
    return datasets
