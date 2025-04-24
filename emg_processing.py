from emg_classes import *
from emg_functions import get_new_filepath
from emg_functions import movements, repetitions, global_fs
from functools import cache


def resample_readings(trial, fs=global_fs, print_verification=True, gestures=movements, reps=repetitions):
    for m in gestures:
        for i in reps:
            new_fpath = get_new_filepath(f'{m}{i}', trial=trial)

            try:
                emg = EMG(f'{m}{i}', version='old', trial=trial, filtered=False)
                emg.resample_EMG(fs=fs)
                data = emg.__array__().T

                with open(new_fpath, 'w') as file:
                    for t, e0, e1 in data:
                        file.write(f'{t} {e0} {e1}\n')
                    if print_verification:
                        print(f"File contents successfully written to {new_fpath}")
            
            except Exception as e:
                print(f"Skipping {m}{i}\n (Testing): {str(e)}\n")

    print("\nResampling complete.")


def get_gesture_vectors(trial, *gestures, **kwargs):
    return [Gesture(g, trial=trial, **kwargs) for g in gestures]


def test_movement_and_plot(movement, trial, gest_vec, normalise=True):

    test_emg = EMG(movement, version='old', trial=trial, filtered=False)
    test_emg.resample_EMG()
    test_emg.filter_EMG()

    result = test_emg.classify_gesture(gest_vec, normalised=normalise)

    other_vec = None
    for g in gest_vec:
        if g.name == result:
            other_vec = g.gesture_vec

    print(result)
    plt.plot(other_vec, label=f'{result}', linewidth=0.7)
    plt.plot(test_emg.vectorise_movement(), label='test', linewidth=0.7)
    plt.legend()
    plt.show()


def show_classifications(trial, gest_list):
    gestures = [g.name for g in gest_list]
    for g in gestures:
        for i in range(5):
            try:
                emg = EMG(f'{g}{i}', trial=trial)
                print(f"Classified {g}{i} as {emg.classify_gesture(gest_list, normalised=True)}.")
            except Exception:
                pass


def plot_gesture_spectrum(trial, gestures=movements[0:6], name=None, show=True, **kwargs):
    gest_list = get_gesture_vectors(trial, *gestures, readings=4)
    plt.figure(name)
    for g in gest_list:
        g.plot(show=False, label=g.name, **kwargs)
        plt.legend()
    plt.tight_layout()
    if show:
        plt.show()


def test_gesture(testfile_name, trial, gest_list):
    test = EMG(testfile_name, trial=trial, version='old')
    test = test.resample_EMG()

    print(test.classify_gesture(gest_list))


def calculate_accuracy(trial, gestures=movements[1:6], mode='calc', total_readings=5, split=0.4, gest_list=None):
    """
    This function calculates the accuracy of the model for a given trial.
    It can either be given a set of vectorised gestures to classify on,
    or it will take the given data and split it.

    Parameters
    ----------
    trial : int
        Trial number.
    gestures : list[str]
        List of names of gestures to be classified.
    mode : str
        Mode of classification. Can take one of 'calc' or 'given'.\n
        'calc' makes the function split the total data into a sample for vectorisation and 
        the rest for classification and measuring accuracy.\n
        'given' allows the user to provide a ready made set of vectorised gestures to classify on.
    total_readings: int
        Is the total number of readings available for each gesture.
    split : float
        Decides the portion of data to be used as the set of vectorised gestures.
    gest_list : list[Gesture] | tuple[Gesture]
        Where the user can provide the ready set of vectorised gestures.

    Returns
    -------
    float
        Returns the accuracy of classification for the given data.

    Examples
    --------
    >>> calculate_accuracy(trial=6, gestures=('rest', 'fist', 'handflexion'), total_readings=10, split=0.3)
    0.94

    Notes
    -----
    - The calculated accuracy depends only on the provided data.
    """

    if mode == 'calc':
        N = int(total_readings * split)
        gest_list = get_gesture_vectors(trial, *gestures, readings=N)

    elif mode == 'given':
        if gest_list == None:
            raise TypeError("Must input an argument for 'gest_list' for the mode 'given'.")
        else:
            gest_list = [g for g in gest_list if g.name in gestures]
    
    else:
        raise TypeError(f"'mode' can only take 'calc' or 'given' as arguments. '{mode}' is not a valid argument.")

    correct = 0
    total = 0
    for g in gestures:
        for i in range(N, total_readings):
            try:
                emg = EMG(f'{g}{i}', trial=trial)
                classification = emg.classify_gesture(gest_list, normalised=True)
                if classification == g:
                    correct += 1
                total += 1
            except Exception:
                pass

    return correct / total


