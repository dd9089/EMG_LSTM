"""
BioRobotics Lab 1 - EMG Signal Processing
==========================================

This module provides functions for processing EMG signals including:
- Filtering (bandpass, notch)
- Rectification and envelope extraction
- Feature extraction
- Power spectral analysis

Author: BioRobotics Course
Updated: 2025
"""

import numpy as np
import pandas as pd
from scipy import signal
from typing import Optional, Tuple
import warnings


def bandpass_filter(data: np.ndarray, 
                    low_freq: float = 20.0, 
                    high_freq: float = 450.0,
                    sample_rate: float = 200.0,
                    order: int = 4) -> np.ndarray:
    """
    Apply a bandpass filter to EMG data.
    
    EMG signals typically contain useful information in the 20-450 Hz range.
    Lower frequencies are often motion artifacts, and higher frequencies
    are usually noise.
    
    Parameters
    ----------
    data : np.ndarray
        Input signal (1D or 2D with time along axis 0)
    low_freq : float
        Low cutoff frequency in Hz
    high_freq : float
        High cutoff frequency in Hz
    sample_rate : float
        Sampling rate in Hz
    order : int
        Filter order
    
    Returns
    -------
    np.ndarray
        Filtered signal
    
    Example
    -------
    >>> raw_emg = np.random.randn(1000)
    >>> filtered = bandpass_filter(raw_emg, low_freq=20, high_freq=450, sample_rate=1000)
    """
    # Normalize frequencies to Nyquist
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Ensure valid frequency range
    if high >= 1.0:
        high = 0.99
        warnings.warn(f"High frequency adjusted to {high * nyquist} Hz (Nyquist limit)")
    
    # Design Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='bandpass')
    
    # Apply filter (filtfilt for zero phase shift)
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return np.apply_along_axis(lambda x: signal.filtfilt(b, a, x), 0, data)


def notch_filter(data: np.ndarray,
                 notch_freq: float = 60.0,
                 quality_factor: float = 30.0,
                 sample_rate: float = 200.0) -> np.ndarray:
    """
    Apply a notch filter to remove power line interference.
    
    In the US, power lines operate at 60 Hz. In Europe and many other
    countries, it's 50 Hz. This interference can contaminate EMG signals.
    
    Parameters
    ----------
    data : np.ndarray
        Input signal
    notch_freq : float
        Frequency to remove (50 or 60 Hz typically)
    quality_factor : float
        Quality factor (higher = narrower notch)
    sample_rate : float
        Sampling rate in Hz
    
    Returns
    -------
    np.ndarray
        Filtered signal
    """
    # Check if notch frequency is valid for this sample rate
    if notch_freq >= sample_rate / 2:
        warnings.warn(f"Notch frequency {notch_freq} Hz exceeds Nyquist. Skipping notch filter.")
        return data
    
    # Design notch filter
    b, a = signal.iirnotch(notch_freq, quality_factor, sample_rate)
    
    # Apply filter
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return np.apply_along_axis(lambda x: signal.filtfilt(b, a, x), 0, data)


def rectify(data: np.ndarray) -> np.ndarray:
    """
    Full-wave rectification of EMG signal.
    
    Rectification takes the absolute value, making all values positive.
    This is the first step in extracting the EMG envelope.
    
    Parameters
    ----------
    data : np.ndarray
        Input signal
    
    Returns
    -------
    np.ndarray
        Rectified signal (all positive values)
    """
    return np.abs(data)


def envelope(data: np.ndarray,
             cutoff_freq: float = 6.0,
             sample_rate: float = 200.0,
             order: int = 4) -> np.ndarray:
    """
    Extract the EMG envelope using rectification and low-pass filtering.
    
    The envelope represents the overall amplitude/intensity of muscle
    activation over time, removing the high-frequency oscillations.
    
    Parameters
    ----------
    data : np.ndarray
        Input EMG signal
    cutoff_freq : float
        Cutoff frequency for envelope smoothing (typically 3-10 Hz)
    sample_rate : float
        Sampling rate in Hz
    order : int
        Filter order
    
    Returns
    -------
    np.ndarray
        EMG envelope
    """
    # Rectify
    rectified = rectify(data)
    
    # Low-pass filter
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    
    if normalized_cutoff >= 1.0:
        normalized_cutoff = 0.99
    
    b, a = signal.butter(order, normalized_cutoff, btype='lowpass')
    
    if data.ndim == 1:
        return signal.filtfilt(b, a, rectified)
    else:
        return np.apply_along_axis(lambda x: signal.filtfilt(b, a, x), 0, rectified)


def rms(data: np.ndarray, window_size: int = 100) -> np.ndarray:
    """
    Calculate the Root Mean Square (RMS) of the EMG signal.
    
    RMS is a common measure of EMG amplitude and is proportional
    to muscle force.
    
    Parameters
    ----------
    data : np.ndarray
        Input signal
    window_size : int
        Size of the sliding window (in samples)
    
    Returns
    -------
    np.ndarray
        RMS values
    """
    # Square the signal
    squared = data ** 2
    
    # Sliding window mean
    window = np.ones(window_size) / window_size
    
    if data.ndim == 1:
        mean_squared = np.convolve(squared, window, mode='same')
    else:
        mean_squared = np.apply_along_axis(
            lambda x: np.convolve(x, window, mode='same'), 0, squared
        )
    
    # Square root
    return np.sqrt(mean_squared)


def compute_features(data: np.ndarray, 
                     sample_rate: float = 200.0) -> dict:
    """
    Compute common EMG features for classification.
    
    Parameters
    ----------
    data : np.ndarray
        Input EMG signal (1D array for single channel)
    sample_rate : float
        Sampling rate in Hz
    
    Returns
    -------
    dict
        Dictionary of feature names and values
    """
    features = {}
    
    # Time domain features
    features['mean'] = np.mean(data)
    features['std'] = np.std(data)
    features['rms'] = np.sqrt(np.mean(data ** 2))
    features['max'] = np.max(np.abs(data))
    features['min'] = np.min(data)
    
    # Mean Absolute Value (MAV)
    features['mav'] = np.mean(np.abs(data))
    
    # Waveform Length (WL) - sum of absolute differences
    features['wl'] = np.sum(np.abs(np.diff(data)))
    
    # Zero Crossing Rate
    zero_crossings = np.where(np.diff(np.signbit(data)))[0]
    features['zcr'] = len(zero_crossings) / len(data)
    
    # Slope Sign Changes
    diff_sign = np.diff(np.sign(np.diff(data)))
    features['ssc'] = np.sum(diff_sign != 0) / len(data)
    
    # Integrated EMG (sum of absolute values)
    features['iemg'] = np.sum(np.abs(data))
    
    # Frequency domain features
    freqs, psd = signal.welch(data, fs=sample_rate, nperseg=min(256, len(data)))
    
    # Mean frequency
    features['mean_freq'] = np.sum(freqs * psd) / np.sum(psd)
    
    # Median frequency
    cumsum = np.cumsum(psd)
    features['median_freq'] = freqs[np.searchsorted(cumsum, cumsum[-1] / 2)]
    
    # Peak frequency
    features['peak_freq'] = freqs[np.argmax(psd)]
    
    # Total power
    features['total_power'] = np.sum(psd)
    
    return features


def power_spectral_density(data: np.ndarray,
                          sample_rate: float = 200.0,
                          nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Power Spectral Density (PSD) of an EMG signal.
    
    PSD shows how the power of the signal is distributed across frequencies.
    
    Parameters
    ----------
    data : np.ndarray
        Input signal
    sample_rate : float
        Sampling rate in Hz
    nperseg : int
        Length of each segment for Welch's method
    
    Returns
    -------
    freqs : np.ndarray
        Frequency values in Hz
    psd : np.ndarray
        Power spectral density values
    """
    freqs, psd = signal.welch(data, fs=sample_rate, nperseg=min(nperseg, len(data)))
    return freqs, psd


def process_emg_pipeline(data: np.ndarray,
                        sample_rate: float = 200.0,
                        bandpass: Tuple[float, float] = (20, 95),
                        notch: float = 60.0,
                        envelope_cutoff: float = 6.0) -> dict:
    """
    Complete EMG processing pipeline.
    
    Parameters
    ----------
    data : np.ndarray
        Raw EMG signal
    sample_rate : float
        Sampling rate in Hz
    bandpass : tuple
        (low, high) frequencies for bandpass filter
    notch : float
        Notch filter frequency (set to 0 to skip)
    envelope_cutoff : float
        Cutoff frequency for envelope extraction
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'raw': Original signal
        - 'filtered': Bandpass filtered signal
        - 'rectified': Rectified signal
        - 'envelope': EMG envelope
        - 'rms': RMS values
    """
    result = {'raw': data.copy()}
    
    # Remove DC offset
    data = data - np.mean(data)
    
    # Bandpass filter
    filtered = bandpass_filter(data, bandpass[0], bandpass[1], sample_rate)
    
    # Notch filter (if specified)
    if notch > 0 and notch < sample_rate / 2:
        filtered = notch_filter(filtered, notch, sample_rate=sample_rate)
    
    result['filtered'] = filtered
    
    # Rectify
    result['rectified'] = rectify(filtered)
    
    # Envelope
    result['envelope'] = envelope(filtered, envelope_cutoff, sample_rate)
    
    # RMS
    window = int(sample_rate * 0.1)  # 100ms window
    result['rms'] = rms(filtered, window)
    
    return result


def segment_data(data: pd.DataFrame,
                timestamps: np.ndarray,
                events: list[Tuple[float, float, str]]) -> dict[str, pd.DataFrame]:
    """
    Segment data based on event timestamps.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with timestamp column
    timestamps : np.ndarray
        Array of timestamps
    events : list
        List of (start_time, end_time, label) tuples
    
    Returns
    -------
    dict
        Dictionary mapping labels to segmented DataFrames
    """
    segments = {}
    
    for start, end, label in events:
        mask = (timestamps >= start) & (timestamps <= end)
        segment = data.loc[mask].copy()
        
        if label not in segments:
            segments[label] = []
        segments[label].append(segment)
    
    return segments


# Convenience function for Myo data
def process_myo_emg(df: pd.DataFrame, 
                    channels: list[str] = None,
                    sample_rate: float = 200.0) -> pd.DataFrame:
    """
    Process Myo EMG data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with EMG columns (EMG_1 through EMG_8)
    channels : list[str], optional
        Which channels to process. Defaults to all EMG channels.
    sample_rate : float
        Sampling rate (Myo is 200 Hz)
    
    Returns
    -------
    pd.DataFrame
        Processed DataFrame with additional columns
    """
    if channels is None:
        channels = [f'EMG_{i}' for i in range(1, 9)]
    
    result = df.copy()
    
    for ch in channels:
        if ch not in df.columns:
            continue
        
        data = df[ch].values.astype(float)
        processed = process_emg_pipeline(data, sample_rate)
        
        result[f'{ch}_filtered'] = processed['filtered']
        result[f'{ch}_envelope'] = processed['envelope']
        result[f'{ch}_rms'] = processed['rms']
    
    return result


if __name__ == "__main__":
    # Demo with synthetic data
    import matplotlib.pyplot as plt
    
    # Generate synthetic EMG
    np.random.seed(42)
    sample_rate = 1000  # Hz
    duration = 2  # seconds
    t = np.arange(0, duration, 1/sample_rate)
    
    # Simulate EMG: noise + motor unit action potentials
    noise = np.random.randn(len(t)) * 0.1
    # Add some simulated muscle activity
    activity = np.zeros_like(t)
    activity[int(0.5*sample_rate):int(1.5*sample_rate)] = 1
    
    # Convolve with motor unit action potential shape
    muap = np.exp(-((np.arange(100) - 50) ** 2) / 200)
    muscle_signal = np.convolve(activity * np.random.randn(len(t)), muap, mode='same')
    
    raw_emg = noise + muscle_signal * 0.5
    
    # Process
    processed = process_emg_pipeline(raw_emg, sample_rate, bandpass=(20, 450))
    features = compute_features(raw_emg, sample_rate)
    
    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    
    axes[0].plot(t, raw_emg)
    axes[0].set_ylabel('Raw EMG')
    axes[0].set_title('EMG Processing Pipeline Demo')
    
    axes[1].plot(t, processed['filtered'])
    axes[1].set_ylabel('Filtered')
    
    axes[2].plot(t, processed['rectified'])
    axes[2].set_ylabel('Rectified')
    
    axes[3].plot(t, processed['envelope'])
    axes[3].set_ylabel('Envelope')
    axes[3].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig('emg_processing_demo.png')
    print("Saved demo plot to emg_processing_demo.png")
    
    print("\nExtracted features:")
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")
