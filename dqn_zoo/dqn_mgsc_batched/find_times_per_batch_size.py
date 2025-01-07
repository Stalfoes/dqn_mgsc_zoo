import glob
from typing import Tuple, Sequence, Any, Optional, Union
from collections import defaultdict


BASE_FILE_PATH = '/home/kapeluck/scratch/dqn_zoo_results/results/mgscdqn_batched_200m_timing/'
EXTENSION = 'mgscdqn_batched_200m_size{meta_batch_size}_{run_id}_{seed}'
ERR = '.err'
OUT = '.out'

ALL_SIZES_GLOB_EXTENSION = EXTENSION.replace('{meta_batch_size}_{run_id}_{seed}', '*')
SPECIFIC_SIZE_GLOB_EXTENSION = EXTENSION.replace('{run_id}_{seed}', '*')


def get_meta_batch_sizes() -> Sequence[str]:
    """Scan the folder for files that match BASE_FILE_PATH + ALL_SIZES_GLOB_EXTENSION and get
        the meta-batch-sizes that have been run.
    """
    prefix_length = len(BASE_FILE_PATH) + len(ALL_SIZES_GLOB_EXTENSION) - 1
    files = glob.glob(BASE_FILE_PATH + ALL_SIZES_GLOB_EXTENSION + OUT)
    size_suffixes = [file[prefix_length:] for file in files]
    underscore_indices = [file.index('_') for file in size_suffixes]
    sizes = [int(file[:i]) for file,i in zip(size_suffixes,underscore_indices)]
    unique_sizes = sorted(set(sizes))
    return unique_sizes
    
def files(meta_batch_size:int) -> Tuple[Sequence[str], Sequence[str]]:
    """Get the err and out files corresponding to a specific meta-batch-size. Seed does not matter.
    """
    err_files = glob.glob(BASE_FILE_PATH + SPECIFIC_SIZE_GLOB_EXTENSION.format(meta_batch_size=meta_batch_size) + ERR)
    out_files = glob.glob(BASE_FILE_PATH + SPECIFIC_SIZE_GLOB_EXTENSION.format(meta_batch_size=meta_batch_size) + OUT)
    return err_files, out_files

def safe_average(numbers:Sequence[Optional[float]]) -> Tuple[float, int]:
    """Calculates an average of a sequence of numbers and ignores None values.
    Returns float('inf') if there are no non-None numbers.
    Returns the average and the number of non-None's.
    """
    summed:Optional[float] = None
    num_floats = 0
    for number in numbers:
        if number is not None:
            if summed is None:
                summed = 0
            summed += number
            num_floats += 1
    if summed is None:
        return float('inf'), 0
    return summed / num_floats, num_floats

def did_run_finish(filepath:str) -> float:
    """Checks to see if an ERR file contains the string "iteration:   5" within it, meaning it finished all 5 iterations. 
    """
    if filepath.endswith(ERR) == False:
        raise ValueError(f"Must pass in an ERR file, but received {filepath=}")
    with open(filepath, 'r') as file:
        for line in file:
            if line.rstrip().split('] ')[1].startswith('iteration:   5'):
                return True
        return False

def printed_time_to_seconds(time_string:str) -> float:
    """Take an input string like HH:MM:SS.##### and return it as the total number of seconds.
    """
    hours, minutes, seconds = time_string.split(':')
    return int(hours) * 60 * 60 + int(minutes) * 60 + float(seconds)

def time_per_1M(filepath:str) -> float:
    """Calculate the average time taken to run 1M steps / 1 iteration. Will take the highest iteration end time
        and subtract the start time of iteration 1 and then divide by the number of complete iterations.
    It will correct for runs that go past midnight.
    """
    if filepath.endswith(ERR) == False:
        raise ValueError(f"Must pass in an ERR file, but received {filepath=}")
    start_iter1_time:float = None
    n_iters:int = None
    end_time:float = None
    with open(filepath, 'r') as file:
        for line in file:
            line = line.rstrip().split()
            if len(line) < 7:
                continue
            if ' '.join(line[4:7]) == 'Training iteration 1.':
                start_iter1_time = printed_time_to_seconds(line[1])
            elif line[4] == 'iteration:':
                n_iters = int(line[5][:-1])
                if n_iters < 1:
                    n_iters = None
                else:
                    end_time = printed_time_to_seconds(line[1])
    if end_time is None or start_iter1_time is None:
        # print(f"WARNING: Could not find a time for {filepath=}. Skipping...\n\t{end_time=}\n\t{start_iter1_time=}")
        return None
    if end_time < start_iter1_time:
        end_time += 24 * 60 * 60
    return (end_time - start_iter1_time) / n_iters

def get_average_time_for_size(meta_batch_size:int) -> float:
    """Gets the average time for 1M steps / 1 iteration for a specific meta-batch-size.
    For each meta-batch-size and seed, calculate the time per 1M steps and then average those numbers.
    """
    err_files, out_files = files(meta_batch_size)
    times = [time_per_1M(file) for file in err_files]
    avg_time, n_seeds = safe_average(times)
    return avg_time, n_seeds

def table_as_string(table:Sequence[Sequence[Any]]) -> str:
    """Return a table converted to a nice, printable string for easier viewing.
    """
    def formatter(thing:Any) -> str:
        if isinstance(thing, int):
            return 'd'
        elif isinstance(thing, float):
            return 'f'
    delimiter = '\t'
    headers = table[0]
    data = table[1:]
    header_lengths = [len(h) for h in headers]
    ret = [delimiter.join(headers)]
    for row in data:
        ret.append(delimiter.join(
            f'{val:<{length}{formatter(val)}}'
            for val,length in zip(row,header_lengths)
        ))
    return '\n'.join(ret)

def average_time_dict_for_size(meta_batch_size:int, average:bool=True) -> dict[str,float]:
    """Returns the average time per call as a dictionary for a given meta-batch-size.
    """
    err_files, out_files = files(meta_batch_size)
    time_dicts = [get_last_time_dict(file) for file in out_files]
    summed_time_dict = sum_dictionaries(time_dicts)
    if average:
        return average_time_per_call(summed_time_dict)
    else:
        return sum_time_dict(summed_time_dict)

def get_keys(time_dict:dict[str,float], keys:Sequence[str]) -> Sequence[str]:
    """Returns a list of values associated with the keys given.
    """
    return [time_dict[key] for key in keys]

def get_last_time_dict(filepath:str) -> dict[str,dict[str,Union[float,int]]]:
    if filepath.endswith(OUT) == False:
        raise ValueError(f"Must pass in an OUT file, but received {filepath=}")
    dict_string_line:str = None
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip().startswith("{'meta-update'"):
                dict_string_line = line
    if dict_string_line is not None:
        return dict_string_to_dict(dict_string_line)
    else:
        return None

def dict_string_to_dict(dict_string:str) -> dict[str,dict[str,Union[float,int]]]:
    """Takes in a string of the timing dict and converts it to a dict."""
    import json
    return json.loads(dict_string.lstrip().replace("'", '"'))

def sum_dictionaries(dictionaries:Sequence[dict[str,dict[str,Union[float,int]]]]) -> dict[str,dict[str,Union[float,int]]]:
    """Returns the sum of a sequence of dictionaries.
    """
    ret = defaultdict(lambda: defaultdict(lambda: 0))
    N = len(dictionaries)
    for dictionary in dictionaries:
        if dictionary is None:
            continue
        for key, inner_dict in dictionary.items():
            for time_key, time_val in inner_dict.items():
                ret[key][time_key] += time_val
    return ret

def average_time_per_call(dictionary:dict[str,dict[str,Union[float,int]]]) -> dict[str,float]:
    """Returns the average time per call from the dictionary.
    """
    ret = defaultdict(lambda: 0.0)
    for key, inner_dict in dictionary.items():
        t = inner_dict['total_time']
        n = inner_dict['num_calls']
        if n != 0:
            ret[key] = t / n
    return ret

def sum_time_dict(dictionary:dict[str,dict[str,Union[float,int]]]) -> dict[str,float]:
    """Returns the total time from all calls from the dictionary.
    """
    ret = defaultdict(lambda: 0.0)
    for key, inner_dict in dictionary.items():
        ret[key] = inner_dict['total_time']
    return ret


if __name__ == '__main__':
    meta_batch_sizes = get_meta_batch_sizes()
    table = [['meta-batch-size', 'n_seeds', 'avg-per-1M (seconds)', 'avg-per-200M (days)']]
    for size in meta_batch_sizes:
        avg_time_1m, num_seeds = get_average_time_for_size(size)
        avg_time_200m_days = (avg_time_1m * 200) / 60 / 60 / 24 
        table.append([size, num_seeds, avg_time_1m, avg_time_200m_days])
    print(table_as_string(table))
    print()

    table = [['meta-batch-size']]
    for size in meta_batch_sizes:
        average_time_dict = average_time_dict_for_size(size, average=False)
        keys = list(average_time_dict.keys())
        if len(table[0]) == 1:
            table[0] += keys
        table.append([size, *get_keys(average_time_dict, keys)])
    print(table_as_string(table))