import matplotlib.pyplot as plt
from typing import TypeVar, Dict, Callable, List, Tuple
import collections
import numpy as np


KeyType = TypeVar('KeyType')
ValueType = TypeVar('ValueType')


def moving_average(values, window_size=10):
    """Code stolen directly from dqn_zoo_plots.ipynb

    Takes in an array and returns an array of the same length that has been smoothed over.
    """
    # numpy.convolve uses zero for initial missing values, so is not suitable.
    numerator = np.nancumsum(values)
    # The sum of the last window_size values.
    numerator[window_size:] = numerator[window_size:] - numerator[:-window_size]
    denominator = np.ones(len(values)) * window_size
    denominator[:window_size] = np.arange(1, window_size + 1)
    smoothed = numerator / denominator
    assert values.shape == smoothed.shape
    return smoothed

def load_data_single_seed_csv(path:str, key:str, value:str) -> Dict[KeyType,ValueType]:
    
    file = open(path, 'r')
    lines = file.readlines()
    file.close()
    indices = {k:i for i,k in enumerate(lines[0].rstrip().split(','))}
    data = {}
    for line in lines[1:]:
        split_line = line.rstrip().split(',')
        key_value = split_line[indices[key]]
        try:
            key_value = float(key_value)
        except:
            pass
        value_value = split_line[indices[value]]
        try:
            value_value = float(value_value)
        except:
            pass
        data[key_value] = value_value
    return data

def average_single_seed_csv_files(path_func:Callable, seeds:List[int], key:str, value:str) -> Dict[KeyType,ValueType]:
    averaged_data = collections.defaultdict(lambda: [])
    for seed in seeds:
        data = load_data_single_seed_csv(path_func(seed), key, value)
        for frame in data.keys():
            averaged_data[frame].append(data[frame])
    for frame in averaged_data.keys():
        averaged_data[frame] = sum(averaged_data[frame]) / len(averaged_data[frame])
    return averaged_data

def load_data_multi_seed_average_csv(path:str, environment:str, key:str, value:str) -> Dict[KeyType,ValueType]:
    """Load data from a CSV file where all the seeds and environments are in a single file.

    This type of file is what DQNZoo was shipped with and what the pre-ran agents are stored like.
    """
    file = open(path, 'r')
    lines = file.readlines()
    file.close()
    indices = {k:i for i,k in enumerate(lines[0].rstrip().split(','))}
    data = collections.defaultdict(lambda: [])
    for line in lines[1:]:
        split_line = line.rstrip().split(',')
        if split_line[indices['environment_name']] != environment:
            continue
        # print(split_line[indices['environment_name']])
        key_value = split_line[indices[key]]
        try:
            key_value = float(key_value)
        except:
            pass
        value_value = split_line[indices[value]]
        try:
            value_value = float(value_value)
        except:
            pass
        data[key_value].append(value_value)
    return {frame:sum(data[frame]) / len(data[frame]) for frame in data}

def data_to_x_y(data:Dict[KeyType,ValueType]) -> Tuple[List[KeyType], List[ValueType]]:
    x = [f / (1e6) for f in data.keys()]
    y = data.values()
    y = moving_average(np.asarray(list(y)))
    return x, y


if __name__ == "__main__":
    environment = 'alien'
    # averaged_data = average_single_seed_csv_files(lambda s: f'./results/dqn/seed_{s}.csv', range(0,5), 'frame', 'eval_episode_return')
    dqn_data = load_data_multi_seed_average_csv('/home/kapeluck/scratch/dqn_zoo_results/dqn.csv', environment, 'frame', 'eval_episode_return')
    prioritized_experience_data = load_data_multi_seed_average_csv('/home/kapeluck/scratch/dqn_zoo_results/prioritized.csv', environment, 'frame', 'eval_episode_return')
    metabatchsize_10_data = average_single_seed_csv_files(lambda s: f'/home/kapeluck/scratch/dqn_zoo_results/results/mgscdqn_batched_100m/{environment}/metasize_10/seed_{s}.csv', range(5), 'frame', 'eval_episode_return')
    metabatchsize_50_data = average_single_seed_csv_files(lambda s: f'/home/kapeluck/scratch/dqn_zoo_results/results/mgscdqn_batched_100m/{environment}/metasize_50/seed_{s}.csv', range(5), 'frame', 'eval_episode_return')
    metabatchsize_100_data = average_single_seed_csv_files(lambda s: f'/home/kapeluck/scratch/dqn_zoo_results/results/mgscdqn_batched_100m/{environment}/metasize_100/seed_{s}.csv', range(5), 'frame', 'eval_episode_return')

    # plt.plot(X, list(averaged_data.values()), label='DQN Luke', color='blue', markeredgecolor='black')
    plt.plot(*data_to_x_y(metabatchsize_10_data), label='MGSCDQN meta-batch-size=10', color='blue', markeredgecolor='black')
    plt.plot(*data_to_x_y(metabatchsize_50_data), label='MGSCDQN meta-batch-size=50', color='green', markeredgecolor='black')
    plt.plot(*data_to_x_y(metabatchsize_100_data), label='MGSCDQN meta-batch-size=100', color='purple', markeredgecolor='black')
    plt.plot(*data_to_x_y(dqn_data), label='DQN', color='red', markeredgecolor='black')
    plt.plot(*data_to_x_y(prioritized_experience_data), label='Prioritized Experience', color='orange', markeredgecolor='black')
    plt.legend()
    plt.xlabel('Frame (millions)')
    plt.ylabel('Episodic Return')
    plt.savefig(f'/home/kapeluck/scratch/dqn_zoo_results/results/mgscdqn_batched_100m/{environment}/dqn_vs_prioritized_vs_mgscdqn.png')
    plt.clf()
    