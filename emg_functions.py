import numpy as np
import config
import random
import os


gestures = ("fingerextension", "fingerspread", "fist", "handextension", "handflexion", "rest", "second", "third", "fourth", "fifth")
unique_gestures = ("fingerspread", "fist", "handextension", "handflexion", "rest")
repetitions = (0,1,2,3,4,'_dynamic')
global_fs = 4700


def get_filepath(movement, trial=1):
    return f"{config.path_to_raw_data}\\Trial{trial}\\EMG_readings_{movement}.txt"


def get_new_filepath(movement, trial=1):
    return f"{config.path_to_resampled_data}\\Trial{trial}\\EMG_readings_{movement}.txt"


def list_files_in_directory(directory_path) -> list[str]:
    try:
        all_entries = os.listdir(directory_path)
        
        files = [f for f in all_entries if os.path.isfile(os.path.join(directory_path, f))]
        
        return files
    except FileNotFoundError:
        print(f"The directory {directory_path} does not exist.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def read_emg_data(filepath):
    values = []

    with open(filepath, "r") as file:
        for line in file:
            read_vals = line.strip().split()

            if len(read_vals) != len(values):
                for i in read_vals:
                    values.append([float(i)])

            else:
                for i in range(len(values)):
                    values[i].append(float(read_vals[i]))

    return tuple(values)


def get_emg_data(movement, version='new', trial=1):
    if version == 'new':
        filepath = get_new_filepath(movement, trial=trial)
    elif version == 'old':
        filepath = get_filepath(movement, trial=trial)
    else:
        raise KeyError("Invalid argument for 'version'. Please pick a valid mode: 'new' or 'old'.")
    
    return read_emg_data(filepath)


def calc_avg_dt(time):
    return np.average(np.diff(time))


def calc_avg_fs(time):
    return 1/calc_avg_dt(time)


def get_sum(*funcs):
    sum = funcs[0]
    for f in funcs[1:]:
        sum = sum + f
    return sum


def get_average(*funcs):
    return get_sum(*funcs) / len(funcs)
    

def normalise(vec):
    return vec / np.linalg.norm(vec)


def get_dist(vec1, vec2, normalised=False):
    if normalised:
        vec1, vec2 = normalise(vec1), normalise(vec2)
    return np.linalg.norm(vec1 - vec2)


def dot_product(vec1, vec2, normalised=False):
    if normalised:
        vec1, vec2 = normalise(vec1), normalise(vec2)
    return np.dot(vec1, vec2)


def random_split(lower, upper, n):
    if upper < lower:
        raise ValueError("Upper bound must be greater than or equal to lower bound.")
    
    full_range = list(range(lower, upper + 1))
    
    if n > len(full_range):
        raise ValueError("n cannot be greater than the total number of integers in the range.")
    
    random_selection = random.sample(full_range, n)
    remaining = [x for x in full_range if x not in random_selection]
    
    return random_selection, remaining
