from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.fft as sp_fft
from scipy.interpolate import interp1d
from emglib.emg_functions import get_emg_data, get_average, dot_product, get_dist
from emglib.emg_functions import global_fs



class Signal:

    def __init__(
        self,
        t : np.ndarray | list[float] | tuple[float],
        val: np.ndarray | list[float] | tuple[float]
    ):

        if len(t) != len(val):
            raise ValueError("Both input arrays must be of the same size.")
        
        self.t = t if isinstance(t, np.ndarray) else np.array(t)
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
        self.val = sp_fft.ifft(X_f).real
        return self
    

    def low_pass_filter(self, cutoff_freq, order=4):
        sampling_rate = self.calc_avg_fs()
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        self.val = signal.filtfilt(b, a, self.val)
        
        return self
    

    def notch_filter(self, freq=50, Q=30, freq_range=2, num_freqs = 1):
        fs = self.calc_avg_fs()

        freqs, magnitudes = self.get_fft()
        freq_mask = (freqs >= freq - freq_range) & (freqs <= freq + freq_range)
        sorted_indices = np.argsort(magnitudes[freq_mask])[-num_freqs:]
        strongest_freqs = freqs[freq_mask][sorted_indices]

        for f in strongest_freqs:
            w0 = f / (fs / 2)
            b, a = signal.iirnotch(w0, Q)
            self.val = signal.filtfilt(b, a, self.val)
        
        return self
    

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
        return self


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
        gesture : str,
        trial : int = 1,
        version : str = 'new',
        filtered : bool = True
    ):
        self.gesture = gesture
        t, *ch = get_emg_data(gesture, version=version, trial=trial)
        t = np.asarray(t)

        self.channels : list[Signal] = [Signal(t, c) for c in ch]
        self.num_channels = len(self.channels)

        if filtered:
            self.filter_EMG()


    def filter_EMG(self, lpf_freq=500, **kwargs):
        self.channels = [c.comb_filter(**kwargs).low_pass_filter(lpf_freq) for c in self.channels]
        return self


    def __array__(self):
        emg_sigs = [s.val for s in self.channels]
        return np.array((self.channels[0].t, *emg_sigs))


    def resample_EMG(self, fs=global_fs):
        self.channels = [c.resample_signal(fs=fs) for c in self.channels]
        return self


    def vectorise_gesture(self, f_range=(0,500)):
        """Turns each gesture into a vector in the frequency domain"""
        ffts = [c.get_fft(spectrum='half', mode='abs', f_range=f_range)[1] for c in self.channels]
        return np.concatenate((ffts))
    

    def classify_gesture(
        self,
        gestures : list[Gesture] | tuple[Gesture],
        normalised : bool = True
    ):

        test = self.vectorise_gesture()

        dot_products = []
        euclidean_distances = []

        for g in gestures:
            dp = dot_product(test, g.gesture_vec, normalised=normalised)
            norm = get_dist(test, g.gesture_vec, normalised=False)
            dot_products.append(dp)
            euclidean_distances.append(1/norm)

        dot_products = np.array(dot_products) / np.max(dot_products)
        euclidean_distances = np.array(euclidean_distances) / np.max(euclidean_distances)

        v = dot_products + euclidean_distances

        max_index = np.argmax(v)
        return gestures[max_index].name


    def plot_channels(self, show=True):

        for i in range(1, self.num_channels+1):
            plt.subplot(self.num_channels,1,i)
            plt.plot(*self.channels[i-1], linewidth=0.7)
            plt.title(f"{self.gesture} Ch{i-1}")
            plt.xlabel("Time (s)")
            plt.ylabel("EMG(t)")

        plt.tight_layout()
        if show:
            plt.show()


    def plot_channel(self, channel, **kwargs):
        try:    
            self.channels[channel].plot(**kwargs)
        except Exception:
            raise ValueError(f"Pick a valid channel number from 0 to {self.num_channels}. Channel {channel} not available.")


    def plot_channels_and_ffts(self, show=True):

        for i in range(1, self.num_channels+1):
            plt.subplot(2,self.num_channels,i)
            plt.plot(*self.channels[i-1], linewidth=0.7)
            plt.title(f"{self.gesture} Ch{i-1}")
            plt.xlabel("Time (s)")
            plt.ylabel("EMG(t)")

            plt.subplot(2,self.num_channels,i+self.num_channels)
            plt.plot(*self.channels[i-1].get_fft(), linewidth=0.7)
            plt.title(f"Ch{i-1} FFT")
            plt.xlabel("Frequency")
            plt.ylabel("Magnitude")

        plt.tight_layout()
        if show:
            plt.show()


    def show_new_EMG(self):
        self.resample_EMG().filter_EMG().plot_channels()




class Gesture:

    def __init__(
        self,
        name : str,
        trial : int = 1,
        version : str = 'new',
        readings: int = 3, 
        reading_indices : list[int] | tuple[int] | None = None, 
        error_message=False
    ):
        
        self.name = name
        self.gesture_vec = self.get_vectorised_gesture(
                                trial=trial,
                                version=version,
                                readings=readings,
                                reading_indices=reading_indices,
                                error_message=error_message
                            )

    def __add__(self, other : Gesture):
        self.gesture_vec = get_average(self.gesture_vec, other.gesture_vec)
        return self
    
    def __array__(self):
        return self.gesture_vec

    def __len__(self):
        return len(self.gesture_vec)

    def get_vectorised_gesture(self, trial=1, version='new', readings=3, reading_indices=None, error_message=False):
        if reading_indices == None:
            if readings:
                reading_indices = range(readings)
            else: raise TypeError("You must provide a value for 'readings'.")

        m_vectors = []
        for i in reading_indices:
            try:
                m_vectors.append(EMG(f'{self.name}{i}', version=version, trial=trial).vectorise_gesture())
            except Exception as e:
                if error_message:
                    print(f"Skipped {self.name}{i}. {e}")
                if reading_indices == None:
                    break
        return get_average(*m_vectors)

    def plot(self, show=True, **kwargs):
        plt.plot(self.gesture_vec, **kwargs)
        if show:
            plt.show()