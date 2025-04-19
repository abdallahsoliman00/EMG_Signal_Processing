import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.fft as sp_fft
from scipy.interpolate import interp1d
from emg_functions import get_emg_data, get_average, dot_product, get_dist
from emg_functions import global_fs
import config


class Signal:

    def __init__(
        self,
        t : np.ndarray | list[float],
        val: np.ndarray | list[float]
    ):

        if len(t) != len(val):
            raise ValueError("Both input arrays must be of the same size.")
        
        self.t = np.array(t)
        self.val = np.array(val)
    

    def __array__(self):
        return np.array((self.t, self.val))


    def __add__(self, other):
        if len(self.t) == len(other.t):
            return Signal(self.t, self.val + other.val)
        else:
            raise ValueError("Signals are not of the same length.")
    

    def __str__(self):
        return f"t = {self.t}\nf(t) = {self.val}"
    

    def __len__(self):
        return len(self.val)
    

    def __iter__(self):
        return iter((self.t, self.val))
    

    def calc_avg_dt(self):
        return np.mean(np.diff(self.t))
    

    def calc_avg_fs(self):
        return 1/self.calc_avg_dt()
    
    
    def remove_dc(self):
        X_f = sp_fft.fft(self.val)
        X_f[0] = 0
        return Signal(self.t, sp_fft.ifft(X_f).real)
    

    def low_pass_filter(self, cutoff_freq, order=4):
        sampling_rate = self.calc_avg_fs()
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        filtered_signal = signal.filtfilt(b, a, self.val)
        
        return Signal(self.t, filtered_signal)
    

    def notch_filter(self, freq=50, Q=30, freq_range=2, num_freqs = 1):
        fs = self.calc_avg_fs()

        freqs, magnitudes = self.get_fft()
        freq_mask = (freqs >= freq - freq_range) & (freqs <= freq + freq_range)
        sorted_indices = np.argsort(magnitudes[freq_mask])[-num_freqs:]
        strongest_freqs = freqs[freq_mask][sorted_indices]

        filtered_signal = self.val
        for f in strongest_freqs:
            w0 = f / (fs / 2)
            b, a = signal.iirnotch(w0, Q)
            filtered_signal = signal.filtfilt(b, a, filtered_signal)
        
        return Signal(self.t, filtered_signal)
    

    def comb_filter(self, fund_freq=50, comb_teeth=12):
        self = self.notch_filter(fund_freq, num_freqs=3)

        if comb_teeth > 1:
            for i in range(2, comb_teeth+1):
                self = self.notch_filter(fund_freq*i)

        return self
    

    def resample_signal(self, fs=global_fs, kind='linear'):

        t_old, x_old = self.t, self.val
        t_min, t_max = t_old[0], t_old[-1]
        t_new = np.arange(t_min, t_max, 1/fs)

        interpolator = interp1d(t_old, x_old, kind=kind, fill_value="extrapolate")
        x_new = interpolator(t_new)

        self.t, self.val = t_new, x_new


    def get_fft(self, spectrum='half', f_range=(0,np.inf), mode='abs', inc_dc=True):
        if mode == 'abs':
            signal_fft = np.abs(sp_fft.fft(self.val))
        elif mode == 'comp':
            signal_fft = sp_fft.fft(self.val)
        else:
            raise KeyError("Invalid argument for 'mode'. Please pick between 'abs' and 'comp'.")
        
        freqs = sp_fft.fftfreq(len(self.val), d=self.calc_avg_dt())

        if spectrum == 'half':
            filt = (freqs > f_range[0]) & (freqs < f_range[1])
            return freqs[filt], signal_fft[filt]
        
        elif spectrum == 'full':
            filt = (freqs > -f_range[1]) & (freqs < f_range[1])
            magnitudes = signal_fft[filt]
            if not inc_dc:
                magnitudes[0] = 0
            return freqs[filt], magnitudes
        
        else:
            raise TypeError("You can only pick between a 'half' and a 'full' spectrum FFT.")

    def plot(self, show=True, **kwargs):
        plt.plot(self.t, self.val, **kwargs)
        if show:
            plt.show()




class EMG:

    def __init__(
        self,
        movement : str,
        version : str = 'new',
        trial : int = 1,
        filtered : bool = True
    ):
        self.movement = movement
        t, ch0, ch1 = get_emg_data(movement, new_old=version, trial=trial)

        self.ch0 = Signal(t, ch0)
        self.ch1 = Signal(t, ch1)

        if filtered:
            self.filter_EMG()


    def filter_EMG(self, **kwargs):
        self.ch0 = self.ch0.comb_filter(**kwargs).low_pass_filter(500)
        self.ch1 = self.ch1.comb_filter(**kwargs).low_pass_filter(500)


    def __array__(self):
        t, emg0 = self.ch0.__array__()
        t, emg1 = self.ch1.__array__()
        return np.array((t, emg0, emg1))


    def resample_EMG(self, fs=global_fs):
        self.ch0.resample_signal(fs=fs)
        self.ch1.resample_signal(fs=fs)
        return self


    # Turns each movement into a vector in the frequency domain
    def vectorise_movement(self, f_range=(0,500)):
        _, fft0 = self.ch0.get_fft(spectrum='half', mode='abs', f_range=f_range)
        _, fft1 = self.ch1.get_fft(spectrum='half', mode='abs', f_range=f_range)
        return np.concatenate((fft0, fft1))
    

    def classify_gesture(
        self,
        gestures : list,
        normalised : bool = True
    ):

        test = self.vectorise_movement()

        dot_products = []
        euclidean_distances = []

        for g in gestures:
            dp = dot_product(test, g.gesture_vec, normalised=normalised)
            norm = get_dist(test, g.gesture_vec, normalised=normalised)
            dot_products.append(dp)
            euclidean_distances.append(1/norm)

        dot_products = np.array(dot_products) / np.max(dot_products)
        euclidean_distances = np.array(euclidean_distances) / np.max(euclidean_distances)

        v = dot_products + euclidean_distances

        max_index = np.argmax(v)
        return gestures[max_index].name


    def plot_channels(self, show=True):

        plt.subplot(2,1,1)
        plt.plot(*self.ch0, linewidth=0.7)
        plt.title(f"{self.movement} Ch0")
        plt.xlabel("Time (s)")
        plt.ylabel("EMG(t)")

        plt.subplot(2,1,2)
        plt.plot(*self.ch1, linewidth=0.7)
        plt.title(f"{self.movement} Ch1")
        plt.xlabel("Time (s)")
        plt.ylabel("EMG(t)")

        plt.tight_layout()
        if show:
            plt.show()


    def plot_channel(self, channel, **kwargs):
        if channel == 0:
            self.ch0.plot(**kwargs)
        elif channel == 1:
            self.ch1.plot(**kwargs)
        else:
            raise ValueError(f"Pick a channel number 0 or 1. Channel number {channel} not available.")


    def plot_channels_and_ffts(self, show=True):

        plt.subplot(2,2,1)
        plt.plot(*self.ch0, linewidth=0.7)
        plt.title(f"{self.movement} Ch0")
        plt.xlabel("Time (s)")
        plt.ylabel("EMG(t)")

        plt.subplot(2,2,2)
        plt.plot(*self.ch1, linewidth=0.7)
        plt.title(f"{self.movement} Ch1")
        plt.xlabel("Time (s)")
        plt.ylabel("EMG(t)")

        plt.subplot(2,2,3)
        plt.plot(*self.ch0.get_fft(), linewidth=0.7)
        plt.title("Ch0 FFT")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")

        plt.subplot(2,2,4)
        plt.plot(*self.ch1.get_fft(), linewidth=0.7)
        plt.title("Ch1 FFT")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")

        plt.tight_layout()
        if show:
            plt.show()




class Gesture:

    def __init__(
        self,
        name : str,
        **kwargs
    ):
        
        self.name = name
        self.gesture_vec = self.get_vectorised_movement(**kwargs)

    def __add__(self, other):
        self.gesture_vec = get_average(self.gesture_vec, other.gesture_vec)
        return self
    
    def __array__(self):
        return self.gesture_vec

    def __len__(self):
        return len(self.gesture_vec)
    
    def plot(self, show=True, **kwargs):
        plt.plot(self.gesture_vec, **kwargs)
        if show:
            plt.show()

    def get_vectorised_movement(self, readings=4, version='new', trial=1, error_message=False):
        m_vectors = []
        for i in range(readings):
            try:
                m_vectors.append(EMG(f'{self.name}{i}', version=version, trial=trial).vectorise_movement())
            except Exception as e:
                if error_message:
                    print(f"Skipped {self.name}{i}. {e}")
                break
        return get_average(*m_vectors)


