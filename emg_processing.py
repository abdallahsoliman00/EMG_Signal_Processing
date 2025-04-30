from emg_classes import *
from emg_functions import get_new_filepath, random_split, get_sum, list_files_in_directory
from emg_functions import gestures, global_fs, unique_gestures
from config import path_to_raw_data
import pandas as pd


def resample_readings(trial, fs=global_fs, print_verification=True):
    """Resamples all raw EMG data and saves it to a new path."""
    filenames = list_files_in_directory(f"{path_to_raw_data}\\Trial{trial}")
    for f in filenames:
        fname = f.removeprefix('EMG_readings_').removesuffix('.txt')
        new_fpath = get_new_filepath(fname, trial=trial)

        try:
            emg = EMG(fname, version='old', trial=trial, filtered=False)
            emg.resample_EMG(fs=fs)
            data = emg.__array__().T

            with open(new_fpath, 'w') as file:
                for row in data:
                    line = ' '.join(str(item) for item in row)
                    file.write(line + '\n')
                if print_verification:
                    print(f"File contents successfully written to {new_fpath}")
        
        except Exception as e:
            print(f"Error encountered: {str(e)}\n")

    print("\nResampling complete.")


def get_gesture_vectors(trial, *gestures, **kwargs):
    """Creates the set of reference vectors for the given gesture names.
    The number of basis vectors and the specific measurements to use can also be specified"""
    return [Gesture(g, trial=trial, **kwargs) for g in gestures]


def test_gesture_and_plot(gesture, trial, gest_vec, normalise=True):

    test_emg = EMG(gesture, version='old', trial=trial, filtered=False)
    test_emg.resample_EMG()
    test_emg.filter_EMG()

    result = test_emg.classify_gesture(gest_vec, normalised=normalise)

    other_vec = None
    for g in gest_vec:
        if g.name == result:
            other_vec = g.gesture_vec

    print(result)
    plt.plot(other_vec, label=f'{result}', linewidth=0.7)
    plt.plot(test_emg.vectorise_gesture(), label='test', linewidth=0.7)
    plt.legend()
    plt.show()


def show_classifications(trial, gest_list, num_readings=5):
    """Prints all classifications for a given trial using a given set of reference vectors."""
    gestures = [g.name for g in gest_list]
    for g in gestures:
        for i in range(num_readings):
            try:
                emg = EMG(f'{g}{i}', trial=trial)
                print(f"Classified {g}{i} as {emg.classify_gesture(gest_list, normalised=True)}.")
            except Exception:
                pass


def plot_gesture_spectrum(trial, gestures=gestures[0:6], readings=4, name=None, show=True, **kwargs):
    gest_list = get_gesture_vectors(trial, *gestures, readings=readings)
    plt.figure(name)
    for g in gest_list:
        g.plot(show=False, label=g.name, **kwargs)
        plt.legend()
    plt.tight_layout()
    if show:
        plt.show()


def test_gesture(testfile_name, trial, gest_list):
    """
    Classifies a test gesture given a set of reference vectors.
    
    Parameters
    ----------
    testfile_name : str
        Name of test gesture
    trial : int
        Trial number or test index
    gest_list : list[Gesture] | tuple[Gesture]
        Set of reference vectors

    Returns
    -------
    None
        Prints the name of the gesture the test was classified as.

    Examples
    --------
    >>> gest_list = get_gesture_vectors(trial=1, gestures=('rest', 'fist', 'handflexion'), readings=3)
    >>> test_gesture('rest2', 1, gest_list)
    rest

    """

    test = EMG(testfile_name, trial=trial, version='old')
    test = test.resample_EMG()

    print(test.classify_gesture(gest_list))


def calculate_accuracy(trial, gestures=unique_gestures, mode='calc', total_readings=5, split=0.4, gest_list=None, error_message=False):
    """
    This function creates a confusion matrix and calculates the accuracy of the model for a given trial.
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
    gest_list : list[Gesture] | tuple[Gesture] | None
        Where the user can provide the ready set of vectorised gestures.
    error_message : bool
        Decides whether error messages are shown when a file is not found

    Returns
    -------
    float
        Returns the accuracy of classification for the given data.

    Examples
    --------
    >>> calculate_accuracy(trial=6, gestures=('rest', 'fist', 'handflexion'), total_readings=10, split=0.3)
    0.95

    >>> gest_list = get_gesture_vectors(trial=3, gestures=('rest', 'fist', 'handflexion'), readings=3)
    >>> calculate_accuracy(trial=3, gestures=('rest', 'fist', 'handflexion'), mode='given', total_readings=5, gest_list=gest_list)
    0.87

    Notes
    -----
    - The output depends only on the provided data.
    """

    if mode == 'calc':
        N = int(total_readings * split)
        vec_nums, test_nums = random_split(lower=0, upper=total_readings-1, n=N)
        gest_list = get_gesture_vectors(trial, *gestures, reading_indices=vec_nums, error_message=error_message)

    elif mode == 'given':
        if gest_list == None:
            raise TypeError("Must input an argument for 'gest_list' for the mode 'given'.")
        else:
            test_nums = range(total_readings)
            gest_list = [g for g in gest_list if g.name in gestures]
    
    else:
        raise TypeError(f"'mode' can only take 'calc' or 'given' as arguments. '{mode}' is not a valid argument.")

    correct = 0
    total = 0
    for g in gestures:
        for i in test_nums:
            try:
                emg = EMG(f'{g}{i}', trial=trial)
                classification = emg.classify_gesture(gest_list, normalised=True)
                correct += (classification == g)
                total += 1
            except Exception:
                pass

    return correct / total


def get_confusion_matrix(trial, gestures=unique_gestures, mode='calc', total_readings=5, split=0.4, gest_list=None, error_message=False):
    """
    This function creates a confusion matrix and calculates the accuracy of the model for a given trial.
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
    error_message : bool
        Decides whether error messages are shown when a file is not found

    Returns
    -------
    pd.DataFrame
        Returns the confusion matrix of classification for the given data.

    Examples
    --------
    >>> calculate_accuracy(trial=6, gestures=('rest', 'fist', 'handflexion'), total_readings=10, split=0.3)
    <pd.DataFrame>
    >>> calculate_accuracy(trial=3, gestures=('rest', 'fist', 'handflexion'), mode='given', total_readings=5, gest_list=gest_list)
    <pd.DataFrame>

    Notes
    -----
    - The output depends only on the provided data.
    """

    if mode == 'calc':
        N = int(total_readings * split)
        vec_nums, test_nums = random_split(lower=0, upper=total_readings-1, n=N)
        gest_list = get_gesture_vectors(trial, *gestures, reading_indices=vec_nums, error_message=error_message)

    elif mode == 'given':
        if gest_list == None:
            raise TypeError("Must input an argument for 'gest_list' for the mode 'given'.")
        else:
            test_nums = range(total_readings)
            gest_list = [g for g in gest_list if g.name in gestures]
    
    else:
        raise TypeError(f"'mode' can only take 'calc' or 'given' as arguments. '{mode}' is not a valid argument.")

    confusion_matrix = pd.DataFrame(
    data=0,
    index=gestures,
    columns=gestures
    )
    
    for g in gestures:
        for i in test_nums:
            try:
                emg = EMG(f'{g}{i}', trial=trial)
                classification = emg.classify_gesture(gest_list, normalised=True)
                confusion_matrix.loc[g, classification] += 1
            except Exception as e:
                if error_message:
                    print(e)

    return confusion_matrix


def accuracy_from_confusion_matrix(cm : pd.DataFrame):
    """Takes in a confusion matrix and outputs the accuracy of classification."""
    correct = cm.values.diagonal().sum()
    total = cm.values.sum()
    accuracy = correct / total if total > 0 else 0
    return accuracy


def show_confusion_matrix(N):
    """Shows the confusion matrix for N/5 samples for each participant."""
    cm_arr = []
    for i in range(2,7):
        try:
            cm = get_confusion_matrix(i, gestures=unique_gestures, total_readings=5, split=N/5)
            cm_arr.append(cm)
        except Exception:
            pass

    result = get_sum(*cm_arr)
    print("\nOverall CM:")
    print(result.to_string())
    print("\nAverage Accuracy: ", accuracy_from_confusion_matrix(result))

