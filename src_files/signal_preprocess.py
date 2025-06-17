import numpy as np

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

    
# more preprocessing funcs