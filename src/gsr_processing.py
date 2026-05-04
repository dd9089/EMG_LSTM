"""
BioRobotics Lab 3 - GSR/EDA Signal Processing
===============================================

This module provides functions for processing Galvanic Skin Response (GSR)
/ Electrodermal Activity (EDA) signals including:
- Low-pass filtering (GSR is <5 Hz, no bandpass/notch needed)
- Tonic/Phasic decomposition (SCL and SCR separation)
- SCR peak detection with amplitude, latency, and rise time
- Feature extraction per condition window
- Complete processing pipeline

Key difference from EMG (see emg_processing.py):
  EMG = high-frequency oscillatory signal -> bandpass(20-450Hz), rectify, envelope
  GSR = slowly varying DC signal -> lowpass(5Hz), tonic/phasic decomposition

Author: BioRobotics Course
Updated: 2026
"""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from typing import Optional, Tuple, Dict, List
import warnings


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def lowpass_filter(data: np.ndarray,
                   cutoff_freq: float = 5.0,
                   sample_rate: float = 250.0,
                   order: int = 4) -> np.ndarray:
    """
    Apply a low-pass filter to GSR data.

    GSR signals are slowly varying (< 5 Hz). This removes high-frequency
    noise while preserving the tonic and phasic components.

    Parameters
    ----------
    data : np.ndarray
        Input GSR signal (1D or 2D with time along axis 0)
    cutoff_freq : float
        Cutoff frequency in Hz (default 5.0, suitable for GSR)
    sample_rate : float
        Sampling rate in Hz
    order : int
        Filter order

    Returns
    -------
    np.ndarray
        Filtered signal
    """
    nyquist = sample_rate / 2
    normalized = cutoff_freq / nyquist

    if normalized >= 1.0:
        normalized = 0.99
        warnings.warn(f"Cutoff adjusted to {normalized * nyquist:.1f} Hz (Nyquist limit)")

    b, a = scipy_signal.butter(order, normalized, btype='lowpass')

    if data.ndim == 1:
        return scipy_signal.filtfilt(b, a, data)
    else:
        return np.apply_along_axis(lambda x: scipy_signal.filtfilt(b, a, x), 0, data)


# ---------------------------------------------------------------------------
# Tonic / Phasic Decomposition
# ---------------------------------------------------------------------------

def decompose_eda(data: np.ndarray,
                  sample_rate: float = 250.0,
                  method: str = "highpass") -> Dict[str, np.ndarray]:
    """
    Decompose EDA signal into tonic (SCL) and phasic (SCR) components.

    The tonic component (Skin Conductance Level) is the slow baseline drift.
    The phasic component (Skin Conductance Responses) contains rapid
    event-related responses driven by sympathetic nervous system activation.

    Two methods available:
    - "highpass": Simple highpass/lowpass split at 0.05 Hz (fast, transparent)
    - "neurokit": Uses neurokit2's cvxEDA algorithm (research-grade)

    Parameters
    ----------
    data : np.ndarray
        GSR signal (should already be low-pass filtered)
    sample_rate : float
        Sampling rate in Hz
    method : str
        Decomposition method ("highpass" or "neurokit")

    Returns
    -------
    dict
        'tonic': SCL component (np.ndarray)
        'phasic': SCR component (np.ndarray)
        'cleaned': cleaned/filtered signal (np.ndarray)
    """
    if method == "neurokit":
        try:
            import neurokit2 as nk
            eda_signals, _info = nk.eda_process(data, sampling_rate=int(sample_rate))
            return {
                'tonic': eda_signals['EDA_Tonic'].values,
                'phasic': eda_signals['EDA_Phasic'].values,
                'cleaned': eda_signals['EDA_Clean'].values,
            }
        except ImportError:
            warnings.warn("neurokit2 not available, falling back to highpass method")
            method = "highpass"
        except Exception as e:
            warnings.warn(f"neurokit2 decomposition failed ({e}), falling back to highpass")
            method = "highpass"

    # Simple highpass/lowpass decomposition
    # Tonic = very low frequency component (< 0.05 Hz)
    nyquist = sample_rate / 2
    cutoff = 0.05 / nyquist

    if cutoff >= 1.0:
        # Sample rate too low for this cutoff — return data as-is
        warnings.warn(f"Sample rate {sample_rate} Hz too low for 0.05 Hz decomposition cutoff")
        return {'tonic': data.copy(), 'phasic': np.zeros_like(data), 'cleaned': data.copy()}

    b, a = scipy_signal.butter(4, cutoff, btype='lowpass')
    tonic = scipy_signal.filtfilt(b, a, data)
    phasic = data - tonic

    return {'tonic': tonic, 'phasic': phasic, 'cleaned': data.copy()}


# ---------------------------------------------------------------------------
# SCR Peak Detection
# ---------------------------------------------------------------------------

def detect_scr_peaks(phasic: np.ndarray,
                     sample_rate: float = 250.0,
                     threshold: float = 0.01,
                     min_distance_sec: float = 1.0) -> Dict:
    """
    Detect Skin Conductance Responses (SCR peaks) in the phasic component.

    An SCR is a transient increase in skin conductance lasting 1-5 seconds,
    caused by sympathetic activation of sweat glands. Each peak has:
    - Onset: where the phasic signal starts rising
    - Peak: maximum amplitude
    - Rise time: onset to peak duration
    - Amplitude: peak value above onset

    Parameters
    ----------
    phasic : np.ndarray
        Phasic (SCR) component of the EDA signal
    sample_rate : float
        Sampling rate in Hz
    threshold : float
        Minimum peak amplitude to count as an SCR
    min_distance_sec : float
        Minimum time between peaks in seconds

    Returns
    -------
    dict
        'peaks_idx': array of peak sample indices
        'amplitudes': peak amplitudes
        'rise_times': time from onset to peak (seconds)
        'onsets_idx': array of onset sample indices
    """
    min_distance = int(min_distance_sec * sample_rate)

    peaks, properties = scipy_signal.find_peaks(
        phasic,
        height=threshold,
        distance=min_distance,
        prominence=threshold * 0.5
    )

    amplitudes = phasic[peaks] if len(peaks) > 0 else np.array([])

    # Find onsets (where phasic starts rising before each peak)
    onsets = []
    rise_times = []
    for peak in peaks:
        onset = peak
        search_start = max(0, peak - int(5 * sample_rate))
        for i in range(peak - 1, search_start, -1):
            if phasic[i] <= 0 or phasic[i] >= phasic[i + 1]:
                onset = i + 1
                break
        onsets.append(onset)
        rise_times.append((peak - onset) / sample_rate)

    return {
        'peaks_idx': peaks,
        'amplitudes': amplitudes,
        'rise_times': np.array(rise_times) if rise_times else np.array([]),
        'onsets_idx': np.array(onsets) if onsets else np.array([]),
    }


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def compute_gsr_features(data: np.ndarray,
                         sample_rate: float = 250.0,
                         condition: str = "") -> Dict:
    """
    Extract GSR features from a segment of data (e.g., one experimental condition).

    Computes both tonic (SCL) and phasic (SCR) features.

    Parameters
    ----------
    data : np.ndarray
        GSR signal segment
    sample_rate : float
        Sampling rate in Hz
    condition : str
        Optional label for this condition window

    Returns
    -------
    dict
        Tonic features: scl_mean, scl_std, scl_min, scl_max, scl_range
        Phasic features: scr_count, scr_rate, scr_amp_mean, scr_amp_max,
                         scr_rise_time_mean
    """
    features = {}
    if condition:
        features['condition'] = condition

    # Filter
    filtered = lowpass_filter(data, sample_rate=sample_rate)

    # Decompose
    components = decompose_eda(filtered, sample_rate, method="neurokit")

    # Tonic features (SCL)
    features['scl_mean'] = np.mean(components['tonic'])
    features['scl_std'] = np.std(components['tonic'])
    features['scl_min'] = np.min(components['tonic'])
    features['scl_max'] = np.max(components['tonic'])
    features['scl_range'] = features['scl_max'] - features['scl_min']

    # Phasic features (SCR)
    scr = detect_scr_peaks(components['phasic'], sample_rate)
    duration_min = len(data) / sample_rate / 60.0

    features['scr_count'] = len(scr['peaks_idx'])
    features['scr_rate'] = features['scr_count'] / duration_min if duration_min > 0 else 0

    if len(scr['amplitudes']) > 0:
        features['scr_amp_mean'] = np.mean(scr['amplitudes'])
        features['scr_amp_max'] = np.max(scr['amplitudes'])
        features['scr_rise_time_mean'] = np.mean(scr['rise_times'])
    else:
        features['scr_amp_mean'] = 0.0
        features['scr_amp_max'] = 0.0
        features['scr_rise_time_mean'] = 0.0

    return features


# ---------------------------------------------------------------------------
# Power Spectral Density
# ---------------------------------------------------------------------------

def power_spectral_density(data: np.ndarray,
                           sample_rate: float = 250.0,
                           nperseg: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Power Spectral Density (PSD) of a GSR signal.

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
    freqs, psd = scipy_signal.welch(data, fs=sample_rate,
                                     nperseg=min(nperseg, len(data)))
    return freqs, psd


# ---------------------------------------------------------------------------
# Complete Pipeline
# ---------------------------------------------------------------------------

def process_gsr_pipeline(data: np.ndarray,
                         sample_rate: float = 250.0) -> Dict:
    """
    Complete GSR processing pipeline.
    Analogous to process_emg_pipeline() in emg_processing.py.

    Pipeline: raw -> lowpass filter -> tonic/phasic decomposition -> SCR peaks

    Parameters
    ----------
    data : np.ndarray
        Raw GSR signal
    sample_rate : float
        Sampling rate in Hz

    Returns
    -------
    dict
        'raw': original signal
        'filtered': low-pass filtered
        'tonic': SCL component
        'phasic': SCR component
        'scr_peaks': detected SCR peak info dict
    """
    result = {'raw': data.copy()}

    result['filtered'] = lowpass_filter(data, sample_rate=sample_rate)

    components = decompose_eda(result['filtered'], sample_rate, method="neurokit")
    result['tonic'] = components['tonic']
    result['phasic'] = components['phasic']

    result['scr_peaks'] = detect_scr_peaks(result['phasic'], sample_rate)

    return result


# ---------------------------------------------------------------------------
# Data Segmentation
# ---------------------------------------------------------------------------

def segment_by_events(data: np.ndarray,
                      timestamps: np.ndarray,
                      events: List[Tuple[float, float, str]]) -> Dict[str, np.ndarray]:
    """
    Segment GSR data by experimental condition events.

    Parameters
    ----------
    data : np.ndarray
        GSR signal
    timestamps : np.ndarray
        Timestamp for each sample
    events : list of (start_time, end_time, label)
        Condition windows

    Returns
    -------
    dict
        label -> data array for each condition
    """
    segments = {}
    for start, end, label in events:
        mask = (timestamps >= start) & (timestamps <= end)
        segments[label] = data[mask]
    return segments


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate synthetic GSR signal
    np.random.seed(42)
    sample_rate = 250  # Hz
    duration = 120     # seconds (2 minutes)
    t = np.arange(0, duration, 1 / sample_rate)
    n = len(t)

    # Tonic component: slow baseline drift (0.01-0.05 Hz)
    tonic = 5.0 + 0.5 * np.sin(2 * np.pi * 0.02 * t) + 0.1 * t / duration

    # Phasic component: simulated SCR events
    phasic = np.zeros(n)
    scr_times = [10, 25, 42, 55, 68, 80, 95, 105]  # seconds
    for scr_t in scr_times:
        idx = int(scr_t * sample_rate)
        if idx < n:
            # SCR shape: fast rise (~2s), slow decay (~5s)
            rise_samples = int(2.0 * sample_rate)
            decay_samples = int(5.0 * sample_rate)
            amplitude = 0.3 + np.random.rand() * 0.5

            for j in range(min(rise_samples, n - idx)):
                phasic[idx + j] += amplitude * (j / rise_samples)
            peak_idx = min(idx + rise_samples, n - 1)
            for j in range(min(decay_samples, n - peak_idx)):
                phasic[peak_idx + j] += amplitude * np.exp(-j / (1.5 * sample_rate))

    # Combine + add noise
    raw_gsr = tonic + phasic + np.random.randn(n) * 0.02

    # Process
    processed = process_gsr_pipeline(raw_gsr, sample_rate)
    features = compute_gsr_features(raw_gsr, sample_rate, condition="demo")

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(t, raw_gsr, linewidth=0.8)
    axes[0].set_ylabel('GSR (raw)')
    axes[0].set_title('GSR Processing Pipeline Demo', fontsize=14, fontweight='bold')

    axes[1].plot(t, processed['filtered'], linewidth=0.8, color='green')
    axes[1].set_ylabel('Filtered')

    axes[2].plot(t, processed['tonic'], linewidth=1.5, color='blue', label='Tonic (SCL)')
    axes[2].legend(loc='upper right')
    axes[2].set_ylabel('SCL')

    axes[3].plot(t, processed['phasic'], linewidth=0.8, color='orange', label='Phasic (SCR)')
    # Mark detected peaks
    peaks = processed['scr_peaks']
    if len(peaks['peaks_idx']) > 0:
        peak_times = t[peaks['peaks_idx']]
        peak_amps = peaks['amplitudes']
        axes[3].scatter(peak_times, peak_amps, color='red', s=50, zorder=5, label='SCR peaks')
    axes[3].legend(loc='upper right')
    axes[3].set_ylabel('SCR')
    axes[3].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig('gsr_processing_demo.png', dpi=150)
    print("Saved demo plot to gsr_processing_demo.png")

    print("\nExtracted features:")
    for name, value in features.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")
