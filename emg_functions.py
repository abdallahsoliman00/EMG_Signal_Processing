import numpy as np


movements = ["fingerextension", "fingerspread", "fist", "handextension", "handflexion", "rest", "index", "middle", "ring", "pinky"]
repetitions = [0,1,2,3,4,'_dynamic']
global_fs = 4700


def get_filepath(movement, trial=1):
    return f"C:\\Users\\abdal\\OneDrive\\Documents\\Southampton\\Year 3\\Part III Project\\EMG_readings\\Trial{trial}\\EMG_readings_{movement}.txt"


def get_new_filepath(movement, trial=1):
    return f"C:\\Users\\abdal\\OneDrive\\Documents\\Southampton\\Year 3\\Part III Project\\EMG_readings_rectified\\Trial{trial}\\EMG_readings_{movement}.txt"


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


def get_emg_data(movement, new_old='new', trial=1):
    if new_old == 'new':
        filepath = get_new_filepath(movement, trial=trial)
    elif new_old == 'old':
        filepath = get_filepath(movement, trial=trial)
    else:
        raise KeyError("Invalid argument for 'new_old'. Please pick a valid mode: 'new' or 'old'.")
    
    values = read_emg_data(filepath)

    return values


def calc_avg_dt(time):
    return np.average(np.diff(time))


def calc_avg_fs(time):
    return 1/calc_avg_dt(time)


def get_sum(*funcs):
    sum = np.zeros_like(funcs[0])
    for f in funcs:
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
