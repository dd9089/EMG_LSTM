"""
BioRobotics Lab - EMG, GSR, and LSL Tools
==========================================

This package provides tools for:
- LSL stream discovery and recording
- Real-time EMG visualization
- EMG signal processing (Lab 1)
- GSR/EDA signal processing (Lab 3)
- Myo Armband interface
- BioRadio direct Python interface (no .NET SDK required)
- Proportional control demonstration
- Stroop test for GSR experiments

Usage:
    from src.lsl_utils import discover_streams, LSLRecorder
    from src.emg_processing import bandpass_filter, envelope, compute_features
    from src.gsr_processing import lowpass_filter, decompose_eda, process_gsr_pipeline
    from src.bioradio import BioRadio, scan_for_bioradio
    from src.visualizer import EMGVisualizer
    from src.proportional_control import ProportionalControlDemo
"""

__version__ = "3.0.0"
__author__ = "BioRobotics Course"

from .lsl_utils import (
    discover_streams,
    find_stream,
    LSLRecorder,
    LSLMarkerStream,
    load_xdf,
    load_csv,
)

from .emg_processing import (
    bandpass_filter,
    notch_filter,
    rectify,
    envelope,
    rms,
    compute_features,
    power_spectral_density,
    process_emg_pipeline,
)

from .gsr_processing import (
    lowpass_filter,
    decompose_eda,
    detect_scr_peaks,
    compute_gsr_features,
    process_gsr_pipeline,
    segment_by_events,
)

from .bioradio import (
    BioRadio,
    scan_for_bioradio,
    find_bioradio_port,
    probe_bioradio_port,
    create_lsl_outlet,
    DeviceConfig,
    ChannelConfig,
    DataSample,
    VALID_SAMPLE_RATES,
)
