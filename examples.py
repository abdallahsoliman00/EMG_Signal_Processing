from emglib.emg_classes import *

# Define gesture names
gestures = ("fingerspread", "fist", "handextension", "handflexion", "rest")


# Define an EMG signal
"""This is the raw unfiltered EMG signal from dataset 6."""
rest = EMG('rest0', trial=6, version='old', filtered=False)

# Resample then filter the signal
"""The signal is resampled at 5000 Hz and filtered using a comb filter with a fundamental frequency of 50 Hz."""
rest.resample_EMG(fs=5000).filter_EMG(fund_freq=50)

# Plot frequency spectra for EMG channels
rest.plot_channels_and_ffts()

# Create a reference vector for a gesture
"""
Creates a labeled reference vector each for the 'rest' and 'fist' gestures.
The reference is an average of the three readings: 'rest0', 'rest1', 'rest2'. The same applies to 'fist'.
"""
rest_ref = Gesture('rest', trial=6, version='new', readings=3)
fist_ref = Gesture('fist', trial=6, version='new', readings=3)

# To see what the vectors look like
rest_ref.plot(show=False, label="rest")
fist_ref.plot(show=False, label="fist")
plt.legend()
plt.show()

# Classify a gesture
test_gesture = EMG('fist4', trial=6)
print(test_gesture.classify_gesture([rest_ref, fist_ref]))


"""More signal proessing can be found in emg_processiing.py"""
