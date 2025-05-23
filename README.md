# EMG_Signal_Processing

This repository contains the code for the EMG processing library.

## Getting started

You will need to install some dependencies for the code to run. These can be installed by running:

```
pip install -r requirements.txt
```

## EMG Data Storage

* All files containing EMG data must be saved in a file with the naming convention: 
    ```
    "Trial{trial number}\EMG_readings_{gesture name}{iteration number}.txt"
    ```

* There must be two separate paths for data storage. One for raw data, and one for the resampled data.
  Both raw and resampled data files have identical names (using the above convention), but are stored in separate folders.
  The paths in ``config.py`` must be changed.

## Data
EMG data can be found in the [EMG Gesture Classification Dataset](https://github.com/abdallahsoliman00/EMG_dataset).

Whatever data is used, it must be formatted as such:
```
<timestamp> <emg_channel_1> <emg_channel_2> ... <emg_channel_n>
0 0.012 0.014 ... 0.025
0.0002 0.013 0.013 ... 0.005
```

## Examples
Some code examples on how to use the library can be found in ``examples.py``.

EMG signal processing using the library can be found in ``emg_processing.py``.

## Notes
* For all functions to work properly, all signals must be resampled to have the same sampling frequency.
