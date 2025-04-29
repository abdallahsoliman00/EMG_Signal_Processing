# EMG_Signal_Processing

This repository contains the code for the EMG processing library.

## Getting started

You will need to install some dependencies for the code to run. These can be installed by running:

```
pip install -r requirements.txt
```

## Data storage convention

* All files containing EMG data must be saved in a file with the naming convention: 
    ```
    "Trial{trial number}\EMG_readings_{gesture name}{iteration number}.txt"
    ```

* There must be two separate paths for data storage. One for raw data, and one for the resampled data.
  Both raw and resampled data files have identical names (using the above convention), but are stored in separate folders.
  The paths in config.py must be changed.

## Data
EMG data is provided in the files "EMG_readings" and "EMG_readings_rectified".

## Examples
Some code examples can be found in examples.py

Notes:
* For all functions to work properly, all signals must be resampled to have the same sampling frequency.
