import matplotlib.pyplot as plt
from typing import TypeVar, Dict, Callable, List
import collections


KeyType = TypeVar('KeyType')
ValueType = TypeVar('ValueType')

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


if __name__ == "__main__":
    # averaged_data = average_single_seed_csv_files(lambda s: f'./results/dqn/seed_{s}.csv', range(0,5), 'frame', 'eval_episode_return')
    generated_data = load_data_multi_seed_average_csv('/home/kapeluck/scratch/dqn_zoo_results/dqn.csv', 'pong', 'frame', 'eval_episode_return')
    reservoir_data = average_single_seed_csv_files(lambda s: f'/home/kapeluck/scratch/dqn_zoo_results/results/dqn_reservoir_200m/seed_{s}.csv', range(5), 'frame', 'eval_episode_return')

    # plt.plot(X, list(averaged_data.values()), label='DQN Luke', color='blue', markeredgecolor='black')
    plt.plot([f / (1e6) for f in reservoir_data.keys()], reservoir_data.values(), label='DQN Reservoir', color='blue', markeredgecolor='black')
    plt.plot([f / (1e6) for f in generated_data.keys()], generated_data.values(), label='DQN FiFo', color='red', markeredgecolor='black')
    plt.legend()
    plt.xlabel('Frame (millions)')
    plt.ylabel('Episodic Return')
    plt.savefig('/home/kapeluck/scratch/dqn_zoo_results/dqn_fifo_vs_reservoir.png')
    plt.clf()
    