"""
BioRobotics Lab 1 - LSL Utilities
=================================

This module provides utilities for working with Lab Streaming Layer (LSL).
LSL is a system for unified collection of measurement time series in 
research experiments.

Key concepts:
- Stream: A source of data (e.g., EMG from Myo, EEG from BioRadio)
- Outlet: Publishes data to the LSL network
- Inlet: Receives data from an LSL stream
- XDF: The file format for storing LSL recordings

Author: BioRobotics Course
Updated: 2025
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pylsl


# ============================================================================
# pylsl API compatibility helpers (parameter names vary between versions)
# ============================================================================

def _resolve_streams(wait_time: float = 1.0):
    """Resolve streams with API version compatibility."""
    try:
        # Try positional first (older API)
        return pylsl.resolve_streams(wait_time)
    except TypeError:
        # Try keyword (newer API)
        return pylsl.resolve_streams(wait_time=wait_time)


def _resolve_byprop(prop: str, value: str, minimum: int = 1, timeout: float = 5.0):
    """Resolve streams by property with API version compatibility."""
    try:
        # Try positional first (older API)
        return pylsl.resolve_byprop(prop, value, minimum, timeout)
    except TypeError:
        # Try keyword (newer API)
        return pylsl.resolve_byprop(prop, value, minimum=minimum, timeout=timeout)


@dataclass
class StreamInfo:
    """Information about an LSL stream."""
    name: str
    type: str
    channel_count: int
    sampling_rate: float
    source_id: str
    hostname: str
    channel_names: list[str] = field(default_factory=list)
    
    @classmethod
    def from_pylsl(cls, info: pylsl.StreamInfo) -> "StreamInfo":
        """Create StreamInfo from pylsl.StreamInfo object."""
        # Extract channel names if available
        channel_names = []
        try:
            desc = info.desc()
            channels = desc.child("channels")
            if not channels.empty():
                ch = channels.child("channel")
                while not ch.empty():
                    label = ch.child_value("label")
                    if label:
                        channel_names.append(label)
                    else:
                        channel_names.append(f"ch_{len(channel_names)+1}")
                    ch = ch.next_sibling("channel")
        except Exception:
            pass
        
        # Fill in default channel names if needed
        while len(channel_names) < info.channel_count():
            channel_names.append(f"ch_{len(channel_names)+1}")
        
        return cls(
            name=info.name(),
            type=info.type(),
            channel_count=info.channel_count(),
            sampling_rate=info.nominal_srate(),
            source_id=info.source_id(),
            hostname=info.hostname(),
            channel_names=channel_names
        )
    
    def __str__(self) -> str:
        return (f"Stream: {self.name} ({self.type})\n"
                f"  Channels: {self.channel_count}\n"
                f"  Sample Rate: {self.sampling_rate} Hz\n"
                f"  Source: {self.source_id}@{self.hostname}")


def discover_streams(timeout: float = 2.0) -> list[StreamInfo]:
    """
    Discover all available LSL streams on the network.
    
    This function scans the local network for any devices or applications
    that are broadcasting LSL streams.
    
    Parameters
    ----------
    timeout : float
        How long to wait for streams (in seconds)
    
    Returns
    -------
    list[StreamInfo]
        List of discovered streams
    
    Example
    -------
    >>> streams = discover_streams()
    >>> for s in streams:
    ...     print(s.name, s.type, s.channel_count)
    """
    print(f"Searching for LSL streams (waiting {timeout}s)...")
    streams = _resolve_streams(timeout)
    
    results = []
    for stream in streams:
        info = StreamInfo.from_pylsl(stream)
        results.append(info)
        print(f"  Found: {info.name} ({info.type})")
    
    if not results:
        print("  No streams found. Make sure your device is connected and streaming.")
    
    return results


def find_stream(name: str = None, stream_type: str = None, 
                timeout: float = 5.0) -> Optional[StreamInfo]:
    """
    Find a specific LSL stream by name or type.
    
    Parameters
    ----------
    name : str, optional
        Name of the stream to find (partial match)
    stream_type : str, optional
        Type of stream (e.g., 'EMG', 'EEG', 'Markers')
    timeout : float
        How long to wait for the stream
    
    Returns
    -------
    StreamInfo or None
        The found stream, or None if not found
    """
    if name:
        print(f"Looking for stream with name containing '{name}'...")
        streams = _resolve_byprop("name", name, timeout=timeout)
    elif stream_type:
        print(f"Looking for stream of type '{stream_type}'...")
        streams = _resolve_byprop("type", stream_type, timeout=timeout)
    else:
        streams = _resolve_streams(timeout)
    
    if streams:
        return StreamInfo.from_pylsl(streams[0])
    return None


class LSLRecorder:
    """
    Record data from one or more LSL streams.
    
    This class provides a simple interface for recording LSL data to
    CSV or XDF files. It handles multiple streams and timestamps.
    
    Example
    -------
    >>> recorder = LSLRecorder()
    >>> recorder.add_stream("Myo")  # Add Myo stream
    >>> recorder.start()
    >>> time.sleep(10)  # Record for 10 seconds
    >>> recorder.stop()
    >>> recorder.save("my_recording.csv")
    """
    
    def __init__(self):
        self.inlets: list[tuple[pylsl.StreamInlet, StreamInfo]] = []
        self.data: dict[str, list] = {}
        self.timestamps: dict[str, list] = {}
        self._recording = False
        self._threads: list[threading.Thread] = []
    
    def add_stream(self, name: str = None, stream_type: str = None, 
                   timeout: float = 5.0) -> bool:
        """
        Add an LSL stream to record.
        
        Parameters
        ----------
        name : str, optional
            Name of stream to add
        stream_type : str, optional
            Type of stream to add
        timeout : float
            How long to wait for stream
        
        Returns
        -------
        bool
            True if stream was added successfully
        """
        # Find the stream
        if name:
            streams = _resolve_byprop("name", name, timeout=timeout)
        elif stream_type:
            streams = _resolve_byprop("type", stream_type, timeout=timeout)
        else:
            print("Error: Must specify name or stream_type")
            return False
        
        if not streams:
            print(f"Could not find stream: {name or stream_type}")
            return False
        
        # Create inlet
        info = StreamInfo.from_pylsl(streams[0])
        inlet = pylsl.StreamInlet(streams[0], max_buflen=360)
        
        self.inlets.append((inlet, info))
        self.data[info.name] = []
        self.timestamps[info.name] = []
        
        print(f"Added stream: {info.name} ({info.channel_count} channels)")
        return True
    
    def start(self):
        """Start recording from all added streams."""
        if self._recording:
            print("Already recording!")
            return
        
        if not self.inlets:
            print("No streams added. Use add_stream() first.")
            return
        
        self._recording = True
        
        # Start a recording thread for each inlet
        for inlet, info in self.inlets:
            thread = threading.Thread(
                target=self._record_stream,
                args=(inlet, info),
                daemon=True
            )
            thread.start()
            self._threads.append(thread)
        
        print("Recording started. Press Ctrl+C or call stop() to end.")
    
    def _record_stream(self, inlet: pylsl.StreamInlet, info: StreamInfo):
        """Internal method to record from a single stream."""
        while self._recording:
            try:
                samples, timestamps = inlet.pull_chunk(timeout=0.1)
                if samples:
                    self.data[info.name].extend(samples)
                    self.timestamps[info.name].extend(timestamps)
            except Exception as e:
                print(f"Error recording {info.name}: {e}")
    
    def stop(self) -> dict[str, pd.DataFrame]:
        """
        Stop recording and return the data.
        
        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary mapping stream names to DataFrames
        """
        self._recording = False
        
        # Wait for threads to finish
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()
        
        # Convert to DataFrames
        results = {}
        for inlet, info in self.inlets:
            if self.data[info.name]:
                df = pd.DataFrame(
                    self.data[info.name],
                    columns=info.channel_names
                )
                df.insert(0, 'timestamp', self.timestamps[info.name])
                results[info.name] = df
                print(f"Recorded {len(df)} samples from {info.name}")
        
        return results
    
    def save(self, filepath: str, stream_name: str = None):
        """
        Save recorded data to a CSV file.
        
        Parameters
        ----------
        filepath : str
            Path to save the file
        stream_name : str, optional
            Which stream to save (if multiple). Saves first if not specified.
        """
        if not self.data:
            print("No data to save!")
            return
        
        # Get the data to save
        if stream_name:
            if stream_name not in self.data:
                print(f"Stream '{stream_name}' not found")
                return
            data = self.data[stream_name]
            timestamps = self.timestamps[stream_name]
            info = next(i for _, i in self.inlets if i.name == stream_name)
        else:
            # Use first stream
            _, info = self.inlets[0]
            data = self.data[info.name]
            timestamps = self.timestamps[info.name]
        
        # Create DataFrame and save
        df = pd.DataFrame(data, columns=info.channel_names)
        df.insert(0, 'timestamp', timestamps)
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} samples to {filepath}")
    
    def clear(self):
        """Clear all recorded data."""
        for name in self.data:
            self.data[name].clear()
            self.timestamps[name].clear()


class LSLMarkerStream:
    """
    Create event markers in LSL.
    
    Markers are used to annotate recordings with events like
    "trial start", "gesture: rock", etc.
    
    Example
    -------
    >>> markers = LSLMarkerStream("MyExperiment")
    >>> markers.push("trial_start")
    >>> # ... do something ...
    >>> markers.push("gesture_rock")
    """
    
    def __init__(self, name: str = "Markers"):
        """Create a marker stream with the given name."""
        info = pylsl.StreamInfo(
            name=name,
            type='Markers',
            channel_count=1,
            nominal_srate=0,  # Irregular rate
            channel_format=pylsl.cf_string,
            source_id=f'{name}_{int(time.time())}'
        )
        self.outlet = pylsl.StreamOutlet(info)
        print(f"Created marker stream: {name}")
    
    def push(self, marker: str):
        """Push a marker string to the stream."""
        self.outlet.push_sample([marker])
        print(f"Marker: {marker}")


# Utility functions for data processing

def load_xdf(filepath: str) -> dict[str, pd.DataFrame]:
    """
    Load an XDF file and return data as DataFrames.
    
    Parameters
    ----------
    filepath : str
        Path to the XDF file
    
    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping stream names to DataFrames
    """
    import pyxdf
    
    data, header = pyxdf.load_xdf(filepath)
    
    results = {}
    for stream in data:
        name = stream['info']['name'][0]
        
        # Get channel names
        try:
            channels = stream['info']['desc'][0]['channels'][0]['channel']
            col_names = [ch['label'][0] for ch in channels]
        except (KeyError, IndexError, TypeError):
            col_names = [f'ch_{i}' for i in range(stream['time_series'].shape[1])]
        
        # Create DataFrame
        df = pd.DataFrame(stream['time_series'], columns=col_names)
        df.insert(0, 'timestamp', stream['time_stamps'])
        results[name] = df
    
    return results


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file recorded from this lab."""
    return pd.read_csv(filepath)


if __name__ == "__main__":
    # Quick demo/test
    print("LSL Utilities Demo")
    print("=" * 50)
    
    print("\n1. Discovering streams...")
    streams = discover_streams(timeout=3.0)
    
    if streams:
        print(f"\n2. Found {len(streams)} stream(s):")
        for s in streams:
            print(f"\n{s}")
    else:
        print("\nNo streams found. To test, you can run:")
        print("  python -m pylsl.examples.SendData")
