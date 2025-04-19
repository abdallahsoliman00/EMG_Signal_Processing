from emg_classes import *
from emg_functions import get_new_filepath
from emg_functions import movements, repetitions, global_fs


def resample_readings(trial, fs=global_fs, print_verification=True, gestures=movements):
    for m in gestures:
        for i in repetitions:
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
    plt.plot(other_vec, label=f'{result}')
    plt.plot(test_emg.vectorise_movement(), label='test')
    plt.legend()
    plt.show()


def show_classifications(trial, gest_list):
    for m in movements[0:6]:
        for i in range(5):
            try:
                emg = EMG(f'{m}{i}', trial=trial)
                print(f"Classified {m}{i} as {emg.classify_gesture(gest_list, normalised=True)}.")
            except Exception:
                pass


def plot_gesture_spectrum(trial, gestures=movements[0:6], name=None, show=True):
    gest_list = get_gesture_vectors(trial, *gestures, readings=4)
    plt.figure(name)
    for g in gest_list:
        g.plot(False, label=g.name)
        plt.legend()
    if show:
        plt.show()


def test_gesture(testfile_name, trial, gest_list):
    test = EMG(testfile_name, trial=trial, version='old')
    test = test.resample_EMG()

    print(test.classify_gesture(gest_list))

plot_gesture_spectrum(4)

# plt.subplot(2,1,1)
# plt.title('Unfiltered')
# EMG('fist_dynamic', filtered=False).plot_channel(0, show=False, linewidth=0.7, color='b')
# plt.subplot(2,1,2)
# plt.title('Filtered')
# EMG('fist_dynamic').plot_channel(0, show=False, linewidth=0.7, color='b')

# plt.tight_layout()
# plt.show()

# plot_gesture_spectrum(1, movements[0:6])

# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)

# test_emgo = EMG('rest0', version='old', filtered=False)
# test_emgn = EMG('rest0', version='new', filtered=False)
# old = test_emgo.ch0.get_fft()
# new = test_emgn.ch0.get_fft()

# ax1.plot(*old, linewidth=0.7, color='#0000FF')
# ax1.set_title('Irregular Sampling')
# ax1.set_ylabel('Magnitude')

# ax2.plot(*new, linewidth=0.7, color='#0000FF')
# ax2.set_title('Regular Sampling')
# ax2.set_xlabel('Frequency')
# ax2.set_ylabel('Magnitude')

# plt.tight_layout()
# plt.show()

# t = EMG('rest0', 'old', 1, False).ch0.t
# plt.plot(np.diff(t), linewidth=0.7, color='b')
# plt.xlabel('Sample Number')
# plt.ylabel('dt')
# plt.show()