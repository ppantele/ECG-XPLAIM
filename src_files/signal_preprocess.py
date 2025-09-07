import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def replace_nan(signal, replace_with=0):
    '''
    Replaces NaN values in the ECG signal with a specified value.

    Args:
        signal (numpy.ndarray): Input ECG signal.
        replace_with (float or int): Value to replace NaNs with (default: 0).

    Returns:
        numpy.ndarray: Signal with NaN values replaced.
    '''
    
    signal[np.isnan(signal)] = replace_with
    return signal


def bandpass_filter(signal: np.ndarray, lowcut=0.5, highcut=40.0, fs=500, order=4) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter to a multi-lead ECG signal.
    Args:
        signal (np.ndarray): ECG array, shape (N, 12) or (N, n_leads)
        lowcut (float): Low cutoff frequency in Hz
        highcut (float): High cutoff frequency in Hz
        fs (float): Sampling frequency in Hz
        order (int): Filter order
    Returns:
        np.ndarray: Filtered ECG, same shape as input
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # Apply along time axis (axis=0), for each lead
    filtered = filtfilt(b, a, signal, axis=0)
    return filtered

def notch_filter(signal: np.ndarray, notch_freq=50.0, fs=500, q=30.0) -> np.ndarray:
    """
    Apply a zero-phase notch (bandstop) filter to remove powerline interference.
    Args:
        signal (np.ndarray): ECG array, shape (N, 12) or (N, n_leads)
        notch_freq (float): Notch frequency in Hz (e.g., 50 or 60)
        fs (float): Sampling frequency in Hz
        q (float): Quality factor
    Returns:
        np.ndarray: Filtered ECG, same shape as input
    """
    w0 = notch_freq / (fs / 2)
    b, a = iirnotch(w0, Q=q)
    filtered = filtfilt(b, a, signal, axis=0)
    return filtered
