"""
BioRobotics Lab 1 - Myo Armband Interface
==========================================

This module provides an LSL interface for the Myo Armband.

Two backends are supported:
1. dl-myo (default): Native Bluetooth - NO DONGLE NEEDED
   - Uses computer's built-in Bluetooth
   - Can connect to specific Myo by MAC address
   - Best for classrooms with multiple Myos
   
2. pyomyo (fallback): Requires the Myo Bluetooth dongle
   - Uses the blue USB dongle that came with the Myo
   - More reliable but requires dongle hardware

Requirements:
    # For dl-myo (recommended):
    pip install dl-myo
    
    # For pyomyo (fallback):
    pip install git+https://github.com/PerlinWarp/pyomyo.git

IMPORTANT: Close MyoConnect before running!

Author: BioRobotics Course
Updated: 2025
"""

import sys
import os
import time
import asyncio
import threading
from dataclasses import dataclass, field
from typing import Optional, List
from queue import Queue

import numpy as np

try:
    import pylsl
    HAS_LSL = True
except ImportError:
    HAS_LSL = False
    print("Warning: pylsl not available. Install with: pip install pylsl")

# Try to import dl-myo (native bluetooth, no dongle)
HAS_DLMYO = False
DLMyoClient = None
EMGMode = None
IMUMode = None
ClassifierMode = None

try:
    from myo import MyoClient
    from myo.types import EMGMode, IMUMode, ClassifierMode, EMGData
    DLMyoClient = MyoClient
    HAS_DLMYO = True
except ImportError:
    pass

# Try to import bleak for scanning (can work without full dl-myo)
HAS_BLEAK = False
try:
    import bleak
    HAS_BLEAK = True
except ImportError:
    pass

# Try to import pyomyo (dongle-based)
HAS_PYOMYO = False
try:
    from pyomyo import Myo as PyoMyo, emg_mode
    HAS_PYOMYO = True
except ImportError:
    pass

# Report available backends
if HAS_DLMYO:
    print("dl-myo available (native Bluetooth - no dongle needed)")
elif HAS_BLEAK:
    print("bleak available (can scan for devices, but dl-myo needed for streaming)")
if HAS_PYOMYO:
    print("pyomyo available (requires Myo dongle)")
if not HAS_DLMYO and not HAS_PYOMYO:
    print("WARNING: No Myo streaming backend available!")
    print("  Install dl-myo:  pip install dl-myo")
    print("  Or pyomyo:       pip install git+https://github.com/PerlinWarp/pyomyo.git")


# ============================================================================
# dl-myo Backend (Native Bluetooth - No Dongle)
# ============================================================================

class DLMyoStreamer:
    """
    Streams Myo data using dl-myo (native Bluetooth).
    
    No dongle required - uses computer's built-in Bluetooth!
    Can connect to specific Myo by MAC address.
    
    Streams:
    - EMG: 8 channels at 200Hz (raw/filtered) or 50Hz (preprocessed)
    - IMU: 10 channels at 50Hz (orientation quaternion + accelerometer + gyroscope)
    """
    
    def __init__(self, stream_name: str = "Myo", mac: str = None, mode: str = "raw",
                 enable_imu: bool = True):
        if not HAS_DLMYO:
            raise ImportError(
                "dl-myo not available.\n"
                "Install with: pip install dl-myo"
            )
        if not HAS_LSL:
            raise ImportError("pylsl not available.")
        
        self.stream_name = stream_name
        self.mac = mac
        self.mode_name = mode
        self.enable_imu = enable_imu
        self.sample_rate = 200 if mode in ["raw", "filtered"] else 50
        self.imu_sample_rate = 50  # IMU is always 50Hz
        
        self.client = None
        self.sample_count = 0
        self.imu_sample_count = 0
        self._running = False
        self._loop = None
        self._thread = None
        self.emg_outlet = None
        self.imu_outlet = None
        
    def _setup_lsl_outlets(self):
        """Create LSL outlets for EMG and IMU data."""
        # EMG outlet (8 channels)
        emg_info = pylsl.StreamInfo(
            name=f"{self.stream_name}_EMG",
            type='EMG',
            channel_count=8,
            nominal_srate=self.sample_rate,
            channel_format=pylsl.cf_float32,
            source_id=f'{self.stream_name}_EMG'
        )
        
        desc = emg_info.desc()
        desc.append_child_value("manufacturer", "Thalmic Labs")
        desc.append_child_value("backend", "dl-myo")
        channels = desc.append_child("channels")
        for i in range(8):
            ch = channels.append_child("channel")
            ch.append_child_value("label", f"EMG_{i+1}")
            ch.append_child_value("unit", "raw")
            ch.append_child_value("type", "EMG")
        
        self.emg_outlet = pylsl.StreamOutlet(emg_info)
        print(f"Created LSL outlet: {self.stream_name}_EMG ({self.sample_rate}Hz, 8ch)")
        
        # IMU outlet (10 channels: quat(4) + accel(3) + gyro(3))
        if self.enable_imu:
            imu_info = pylsl.StreamInfo(
                name=f"{self.stream_name}_IMU",
                type='IMU',
                channel_count=10,
                nominal_srate=self.imu_sample_rate,
                channel_format=pylsl.cf_float32,
                source_id=f'{self.stream_name}_IMU'
            )
            
            desc = imu_info.desc()
            desc.append_child_value("manufacturer", "Thalmic Labs")
            desc.append_child_value("backend", "dl-myo")
            channels = desc.append_child("channels")
            
            # Quaternion (orientation)
            for label in ["quat_w", "quat_x", "quat_y", "quat_z"]:
                ch = channels.append_child("channel")
                ch.append_child_value("label", label)
                ch.append_child_value("unit", "normalized")
                ch.append_child_value("type", "Orientation")
            
            # Accelerometer
            for label in ["accel_x", "accel_y", "accel_z"]:
                ch = channels.append_child("channel")
                ch.append_child_value("label", label)
                ch.append_child_value("unit", "g")
                ch.append_child_value("type", "Accelerometer")
            
            # Gyroscope
            for label in ["gyro_x", "gyro_y", "gyro_z"]:
                ch = channels.append_child("channel")
                ch.append_child_value("label", label)
                ch.append_child_value("unit", "deg/s")
                ch.append_child_value("type", "Gyroscope")
            
            self.imu_outlet = pylsl.StreamOutlet(imu_info)
            print(f"Created LSL outlet: {self.stream_name}_IMU ({self.imu_sample_rate}Hz, 10ch)")
    
    def _create_client_class(self):
        """Create a MyoClient subclass with EMG and IMU callbacks that push to LSL."""
        streamer = self  # Capture reference for the callbacks
        
        class LSLMyoClient(DLMyoClient):
            async def on_emg_data(self, emg):
                """Called when EMG data is received."""
                try:
                    # EMGData has sample1 and sample2 attributes
                    for sample in [emg.sample1, emg.sample2]:
                        if sample:
                            streamer.emg_outlet.push_sample(list(sample))
                            streamer.sample_count += 1
                except Exception as e:
                    # Try alternate data format
                    try:
                        if hasattr(emg, '__iter__'):
                            streamer.emg_outlet.push_sample(list(emg)[:8])
                            streamer.sample_count += 1
                    except:
                        pass
            
            async def on_imu_data(self, imu):
                """Called when IMU data is received."""
                if not streamer.enable_imu or streamer.imu_outlet is None:
                    return
                
                try:
                    # IMU data typically has orientation, accelerometer, gyroscope
                    # Format: [quat_w, quat_x, quat_y, quat_z, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
                    sample = []
                    
                    # Orientation (quaternion)
                    if hasattr(imu, 'orientation'):
                        ori = imu.orientation
                        if hasattr(ori, 'w'):
                            sample.extend([ori.w, ori.x, ori.y, ori.z])
                        elif hasattr(ori, '__iter__'):
                            sample.extend(list(ori)[:4])
                        else:
                            sample.extend([1.0, 0.0, 0.0, 0.0])  # Default quaternion
                    elif hasattr(imu, 'quat'):
                        q = imu.quat
                        if hasattr(q, 'w'):
                            sample.extend([q.w, q.x, q.y, q.z])
                        else:
                            sample.extend(list(q)[:4])
                    else:
                        sample.extend([1.0, 0.0, 0.0, 0.0])
                    
                    # Accelerometer
                    if hasattr(imu, 'accelerometer'):
                        acc = imu.accelerometer
                        if hasattr(acc, 'x'):
                            sample.extend([acc.x, acc.y, acc.z])
                        elif hasattr(acc, '__iter__'):
                            sample.extend(list(acc)[:3])
                        else:
                            sample.extend([0.0, 0.0, 0.0])
                    elif hasattr(imu, 'accel'):
                        acc = imu.accel
                        if hasattr(acc, 'x'):
                            sample.extend([acc.x, acc.y, acc.z])
                        else:
                            sample.extend(list(acc)[:3])
                    else:
                        sample.extend([0.0, 0.0, 0.0])
                    
                    # Gyroscope
                    if hasattr(imu, 'gyroscope'):
                        gyro = imu.gyroscope
                        if hasattr(gyro, 'x'):
                            sample.extend([gyro.x, gyro.y, gyro.z])
                        elif hasattr(gyro, '__iter__'):
                            sample.extend(list(gyro)[:3])
                        else:
                            sample.extend([0.0, 0.0, 0.0])
                    elif hasattr(imu, 'gyro'):
                        gyro = imu.gyro
                        if hasattr(gyro, 'x'):
                            sample.extend([gyro.x, gyro.y, gyro.z])
                        else:
                            sample.extend(list(gyro)[:3])
                    else:
                        sample.extend([0.0, 0.0, 0.0])
                    
                    # Push the 10-channel sample
                    if len(sample) == 10:
                        streamer.imu_outlet.push_sample(sample)
                        streamer.imu_sample_count += 1
                
                except Exception as e:
                    # Debug: print the IMU data structure
                    if streamer.imu_sample_count == 0:
                        print(f"IMU data format: {type(imu)}, attrs: {dir(imu)}")
            
            # Required abstract method stubs
            async def on_classifier_event(self, ce):
                pass
            
            async def on_aggregated_data(self, ad):
                pass
            
            async def on_emg_data_aggregated(self, emg):
                pass
            
            async def on_fv_data(self, fvd):
                pass
            
            async def on_motion_event(self, me):
                pass
        
        return LSLMyoClient
    
    async def _run_async(self):
        """Async run loop for dl-myo."""
        print("=" * 50)
        print("Myo LSL Streamer (dl-myo)")
        print("=" * 50)
        print("\nUsing native Bluetooth - no dongle needed!")
        print("IMPORTANT: Make sure MyoConnect is CLOSED!\n")
        
        # Setup LSL outlets
        self._setup_lsl_outlets()
        
        # Create custom client class with our EMG callback
        LSLMyoClient = self._create_client_class()
        
        # Connect to Myo
        if self.mac:
            print(f"Connecting to Myo at {self.mac}...")
            self.client = await LSLMyoClient.with_device(mac=self.mac)
        else:
            print("Scanning for Myo devices...")
            self.client = await LSLMyoClient.with_device()
        
        if self.client is None:
            print("ERROR: Could not find/connect to Myo device!")
            return
        
        print("Connected to Myo!")
        
        # Determine EMG mode
        if self.mode_name == "raw":
            emg_mode = EMGMode.SEND_RAW
        elif self.mode_name == "filtered":
            emg_mode = EMGMode.SEND_FILT
        else:  # preprocessed
            emg_mode = EMGMode.SEND_EMG  # rectified/preprocessed
        
        # Determine IMU mode
        imu_mode = IMUMode.SEND_DATA if self.enable_imu else IMUMode.NONE
        
        # Setup the client
        try:
            await self.client.setup(
                classifier_mode=ClassifierMode.DISABLED,
                emg_mode=emg_mode,
                imu_mode=imu_mode,
            )
        except Exception as e:
            print(f"Warning: Could not setup modes: {e}")
        
        # Start receiving data
        await self.client.start()
        
        # Vibrate to indicate connection
        try:
            from myo.types import VibrationType
            await self.client.vibrate(VibrationType.SHORT)
        except Exception as e:
            print(f"Warning: Could not vibrate: {e}")
        
        print("Myo streamer started!")
        print(f"Streaming EMG at {self.sample_rate}Hz ({self.mode_name} mode)")
        if self.enable_imu:
            print(f"Streaming IMU at {self.imu_sample_rate}Hz (orientation + accel + gyro)")
        
        # Keep running
        self._running = True
        while self._running:
            await asyncio.sleep(0.1)
        
        # Cleanup
        try:
            await self.client.stop()
            await self.client.disconnect()
        except:
            pass
    
    def _run_in_thread(self):
        """Run the async loop in a separate thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_async())
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self._loop.close()
    
    def start(self):
        """Start streaming."""
        self._thread = threading.Thread(target=self._run_in_thread, daemon=True)
        self._thread.start()
        # Wait a bit for connection
        time.sleep(3)
    
    def stop(self):
        """Stop streaming."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        print(f"\nStreamed {self.sample_count} EMG samples")
        if self.enable_imu:
            print(f"Streamed {self.imu_sample_count} IMU samples")
        print("Myo streamer stopped")
    
    @property
    def is_connected(self) -> bool:
        return self._running


# ============================================================================
# pyomyo Backend (Dongle-based)
# ============================================================================

class PyoMyoStreamer:
    """
    Streams Myo data using pyomyo (requires dongle).
    
    Uses the blue Myo Bluetooth dongle via serial port.
    Note: IMU not supported with pyomyo backend.
    """
    
    def __init__(self, stream_name: str = "Myo", mode: str = "raw", tty: str = None, **kwargs):
        # Note: enable_imu and other kwargs are ignored - pyomyo only supports EMG
        if not HAS_PYOMYO:
            raise ImportError(
                "pyomyo not available.\n"
                "Install with: pip install git+https://github.com/PerlinWarp/pyomyo.git"
            )
        if not HAS_LSL:
            raise ImportError("pylsl not available.")
        
        self.stream_name = stream_name
        self.mode_name = mode
        self.tty = tty
        
        if mode == "raw":
            self.emg_mode = emg_mode.RAW
            self.sample_rate = 200
        elif mode == "filtered":
            self.emg_mode = emg_mode.FILTERED
            self.sample_rate = 200
        else:
            self.emg_mode = emg_mode.PREPROCESSED
            self.sample_rate = 50
        
        self.myo = None
        self.sample_count = 0
        self._running = False
        self._thread = None
        self.emg_outlet = None
        self._emg_queue = Queue()
    
    def _setup_lsl_outlet(self):
        """Create LSL outlet."""
        emg_info = pylsl.StreamInfo(
            name=f"{self.stream_name}_EMG",
            type='EMG',
            channel_count=8,
            nominal_srate=self.sample_rate,
            channel_format=pylsl.cf_int8 if self.emg_mode == emg_mode.RAW else pylsl.cf_float32,
            source_id=f'{self.stream_name}_EMG'
        )
        
        desc = emg_info.desc()
        desc.append_child_value("manufacturer", "Thalmic Labs")
        desc.append_child_value("backend", "pyomyo")
        channels = desc.append_child("channels")
        for i in range(8):
            ch = channels.append_child("channel")
            ch.append_child_value("label", f"EMG_{i+1}")
        
        self.emg_outlet = pylsl.StreamOutlet(emg_info)
        print(f"Created LSL outlet: {self.stream_name}_EMG ({self.sample_rate}Hz)")
    
    def _emg_callback(self, emg, movement):
        """Called when EMG data is received."""
        self._emg_queue.put(list(emg))
    
    def _lsl_thread(self):
        """Push EMG data to LSL."""
        while self._running:
            try:
                emg = self._emg_queue.get(timeout=0.1)
                self.emg_outlet.push_sample(emg)
                self.sample_count += 1
            except:
                pass
    
    def _run_loop(self):
        """Main pyomyo run loop."""
        try:
            while self._running:
                self.myo.run()
        except Exception as e:
            print(f"Error: {e}")
            self._running = False
    
    def start(self):
        """Start streaming."""
        print("=" * 50)
        print("Myo LSL Streamer (pyomyo)")
        print("=" * 50)
        print("\nUsing Myo Bluetooth dongle")
        print("IMPORTANT: Make sure MyoConnect is CLOSED!\n")
        
        if self.tty:
            print(f"Connecting via {self.tty}...")
        else:
            print("Auto-discovering dongle...")
        
        self._setup_lsl_outlet()
        
        # Create Myo
        if self.tty:
            self.myo = PyoMyo(mode=self.emg_mode, tty=self.tty)
        else:
            self.myo = PyoMyo(mode=self.emg_mode)
        
        self.myo.connect()
        self.myo.add_emg_handler(self._emg_callback)
        
        self._running = True
        
        # Start threads
        threading.Thread(target=self._lsl_thread, daemon=True).start()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        print("Myo streamer started!")
        print(f"Streaming EMG at {self.sample_rate}Hz ({self.mode_name} mode)")
    
    def stop(self):
        """Stop streaming."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.myo:
            try:
                self.myo.disconnect()
            except:
                pass
        print(f"\nStreamed {self.sample_count} EMG samples")
        print("Myo streamer stopped")
    
    @property
    def is_connected(self) -> bool:
        return self._running


# ============================================================================
# Mock Streamer (for testing)
# ============================================================================

class MockMyoStreamer:
    """Mock Myo streamer for testing without hardware."""
    
    def __init__(self, stream_name: str = "MockMyo", sample_rate: int = 200, enable_imu: bool = True):
        if not HAS_LSL:
            raise ImportError("pylsl not available.")
        
        self.stream_name = stream_name
        self.sample_rate = sample_rate
        self.imu_sample_rate = 50
        self.enable_imu = enable_imu
        self.sample_count = 0
        self.imu_sample_count = 0
        self._running = False
        self._thread = None
        self._imu_thread = None
        self.emg_outlet = None
        self.imu_outlet = None
    
    def _setup_lsl_outlets(self):
        """Create LSL outlets."""
        # EMG outlet
        emg_info = pylsl.StreamInfo(
            name=f"{self.stream_name}_EMG",
            type='EMG',
            channel_count=8,
            nominal_srate=self.sample_rate,
            channel_format=pylsl.cf_float32,
            source_id=f'{self.stream_name}_EMG'
        )
        self.emg_outlet = pylsl.StreamOutlet(emg_info)
        print(f"Created mock LSL outlet: {self.stream_name}_EMG ({self.sample_rate}Hz, 8ch)")
        
        # IMU outlet
        if self.enable_imu:
            imu_info = pylsl.StreamInfo(
                name=f"{self.stream_name}_IMU",
                type='IMU',
                channel_count=10,
                nominal_srate=self.imu_sample_rate,
                channel_format=pylsl.cf_float32,
                source_id=f'{self.stream_name}_IMU'
            )
            self.imu_outlet = pylsl.StreamOutlet(imu_info)
            print(f"Created mock LSL outlet: {self.stream_name}_IMU ({self.imu_sample_rate}Hz, 10ch)")
    
    def start(self):
        """Start generating mock data."""
        print("=" * 50)
        print("Mock Myo Streamer")
        print("=" * 50)
        print("Generating synthetic EMG + IMU data for testing\n")
        
        self._setup_lsl_outlets()
        self._running = True
        self._thread = threading.Thread(target=self._generate_emg_data, daemon=True)
        self._thread.start()
        
        if self.enable_imu:
            self._imu_thread = threading.Thread(target=self._generate_imu_data, daemon=True)
            self._imu_thread.start()
        
        print("Mock streamer started")
    
    def _generate_emg_data(self):
        """Generate synthetic EMG data."""
        t = 0
        dt = 1.0 / self.sample_rate
        while self._running:
            emg = []
            for ch in range(8):
                val = np.random.normal(0, 5)
                if np.sin(2 * np.pi * 0.3 * t + ch * np.pi / 4) > 0.6:
                    val += np.random.normal(50, 20)
                val += 3 * np.sin(2 * np.pi * 60 * t)
                emg.append(val)
            
            self.emg_outlet.push_sample(emg)
            self.sample_count += 1
            t += dt
            time.sleep(dt)
    
    def _generate_imu_data(self):
        """Generate synthetic IMU data."""
        t = 0
        dt = 1.0 / self.imu_sample_rate
        
        # Initial orientation (unit quaternion)
        angle = 0
        
        while self._running:
            # Simulate slow rotation (quaternion)
            angle = (t * 0.5) % (2 * np.pi)  # Slow rotation
            quat_w = np.cos(angle / 2)
            quat_x = 0
            quat_y = np.sin(angle / 2)  # Rotation around Y axis
            quat_z = 0
            
            # Accelerometer (gravity + small movements)
            accel_x = np.random.normal(0, 0.1)
            accel_y = np.random.normal(0, 0.1)
            accel_z = np.random.normal(-1.0, 0.1)  # Gravity pointing down
            
            # Gyroscope (small angular velocities)
            gyro_x = np.random.normal(0, 5)
            gyro_y = np.random.normal(0, 5) + 30 * np.sin(2 * np.pi * 0.2 * t)  # Simulate movement
            gyro_z = np.random.normal(0, 5)
            
            imu_sample = [quat_w, quat_x, quat_y, quat_z, 
                         accel_x, accel_y, accel_z,
                         gyro_x, gyro_y, gyro_z]
            
            self.imu_outlet.push_sample(imu_sample)
            self.imu_sample_count += 1
            t += dt
            time.sleep(dt)
    
    def stop(self):
        """Stop generating data."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._imu_thread:
            self._imu_thread.join(timeout=1.0)
        print(f"\nMock streamer stopped.")
        print(f"  Generated {self.sample_count} EMG samples")
        if self.enable_imu:
            print(f"  Generated {self.imu_sample_count} IMU samples")
    
    @property
    def is_connected(self) -> bool:
        return self._running


# ============================================================================
# Unified Streamer Factory
# ============================================================================

def create_streamer(backend: str = "auto", **kwargs):
    """
    Create a Myo streamer with the specified backend.
    
    Parameters
    ----------
    backend : str
        "dl-myo" - Native Bluetooth (no dongle)
        "pyomyo" - Dongle-based
        "mock"   - Fake data for testing
        "auto"   - Try dl-myo first, fall back to pyomyo
    **kwargs
        Additional arguments passed to the streamer
    
    Returns
    -------
    Streamer object
    """
    if backend == "mock":
        return MockMyoStreamer(**kwargs)
    
    if backend == "dl-myo":
        if not HAS_DLMYO:
            raise ImportError("dl-myo not available. Install with: pip install dl-myo")
        return DLMyoStreamer(**kwargs)
    
    if backend == "pyomyo":
        if not HAS_PYOMYO:
            raise ImportError("pyomyo not available. Install with: pip install git+https://github.com/PerlinWarp/pyomyo.git")
        return PyoMyoStreamer(**kwargs)
    
    # Auto mode
    if HAS_DLMYO:
        print("Using dl-myo backend (native Bluetooth)")
        return DLMyoStreamer(**kwargs)
    elif HAS_PYOMYO:
        print("Using pyomyo backend (dongle-based)")
        return PyoMyoStreamer(**kwargs)
    else:
        raise ImportError(
            "No Myo backend available!\n"
            "Install dl-myo:  pip install dl-myo\n"
            "Or pyomyo:       pip install git+https://github.com/PerlinWarp/pyomyo.git"
        )


# ============================================================================
# Scanning for Devices
# ============================================================================

async def scan_for_myos(timeout: float = 5.0) -> list:
    """Scan for Myo devices using Bluetooth."""
    if not HAS_BLEAK:
        print("bleak required for scanning. Install with: pip install bleak")
        print("(Or install dl-myo which includes bleak: pip install dl-myo)")
        return []
    
    print(f"\nScanning for Myo devices ({timeout}s)...")
    print("-" * 40)
    
    devices = await bleak.BleakScanner.discover(timeout=timeout)
    
    myos = []
    for d in devices:
        # Myo has a specific service UUID or name containing "Myo"
        name = d.name or ""
        if "Myo" in name:
            myos.append({"name": name, "mac": d.address, "rssi": d.rssi})
            rssi_str = f"{d.rssi} dBm" if d.rssi else "N/A"
            print(f"  [{len(myos)}] {name}")
            print(f"      MAC: {d.address}")
            print(f"      Signal: {rssi_str}")
    
    print("-" * 40)
    
    if not myos:
        print("No Myo devices found.")
        print("\nTroubleshooting:")
        print("  - Wake up the Myo by moving/shaking it")
        print("  - Make sure Bluetooth is enabled")
        print("  - Close MyoConnect if it's running")
        print("  - Move closer to the computer")
    else:
        print(f"Found {len(myos)} Myo device(s)")
    
    return myos


async def ping_myo(mac: str, timeout: float = 5.0) -> bool:
    """
    Ping a Myo device to verify it's reachable and identify it.
    The Myo will vibrate when pinged successfully.
    """
    if not HAS_DLMYO:
        print("dl-myo required for pinging. Install with: pip install dl-myo")
        return False
    
    print(f"\nPinging Myo at {mac}...")
    
    try:
        # Create a minimal client just to connect and vibrate
        class PingClient(DLMyoClient):
            async def on_emg_data(self, emg): pass
            async def on_classifier_event(self, ce): pass
            async def on_aggregated_data(self, ad): pass
            async def on_emg_data_aggregated(self, emg): pass
            async def on_fv_data(self, fvd): pass
            async def on_imu_data(self, imu): pass
            async def on_motion_event(self, me): pass
        
        client = await asyncio.wait_for(
            PingClient.with_device(mac=mac),
            timeout=timeout
        )
        
        if client is None:
            print(f"  ✗ Could not connect to {mac}")
            return False
        
        # Vibrate to identify
        try:
            from myo.types import VibrationType
            await client.vibrate(VibrationType.SHORT)
            await asyncio.sleep(0.3)
            await client.vibrate(VibrationType.SHORT)
        except Exception as e:
            print(f"  Warning: Could not vibrate: {e}")
        
        # Disconnect
        try:
            await client.disconnect()
        except:
            pass
        
        print(f"  ✓ Myo at {mac} responded! (It should have vibrated twice)")
        return True
        
    except asyncio.TimeoutError:
        print(f"  ✗ Timeout connecting to {mac}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


async def interactive_select() -> str:
    """
    Interactive mode: scan for Myos, let user select one, and optionally ping it.
    Returns the selected MAC address or None.
    """
    print("=" * 50)
    print("Myo Device Selection")
    print("=" * 50)
    
    # Scan for devices
    myos = await scan_for_myos(timeout=5.0)
    
    if not myos:
        return None
    
    # If only one device, offer to use it directly
    if len(myos) == 1:
        mac = myos[0]['mac']
        print(f"\nOnly one Myo found: {mac}")
        response = input("Use this device? [Y/n]: ").strip().lower()
        if response in ['', 'y', 'yes']:
            # Offer to ping
            ping = input("Ping to verify? (Myo will vibrate) [Y/n]: ").strip().lower()
            if ping in ['', 'y', 'yes']:
                await ping_myo(mac)
            return mac
        return None
    
    # Multiple devices - let user choose
    while True:
        print("\nOptions:")
        print("  [1-{}] Select a Myo by number".format(len(myos)))
        print("  [p #]  Ping a Myo (e.g., 'p 1' to ping device 1)")
        print("  [r]    Rescan for devices")
        print("  [q]    Quit")
        
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == 'q':
            return None
        
        if choice == 'r':
            myos = await scan_for_myos(timeout=5.0)
            if not myos:
                return None
            continue
        
        if choice.startswith('p '):
            try:
                idx = int(choice[2:]) - 1
                if 0 <= idx < len(myos):
                    await ping_myo(myos[idx]['mac'])
                else:
                    print("Invalid device number")
            except ValueError:
                print("Invalid input. Use 'p 1' to ping device 1")
            continue
        
        # Try to parse as device number
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(myos):
                mac = myos[idx]['mac']
                print(f"\nSelected: {myos[idx]['name']} ({mac})")
                
                # Offer to ping
                ping = input("Ping to verify? (Myo will vibrate) [Y/n]: ").strip().lower()
                if ping in ['', 'y', 'yes']:
                    success = await ping_myo(mac)
                    if not success:
                        retry = input("Ping failed. Use anyway? [y/N]: ").strip().lower()
                        if retry not in ['y', 'yes']:
                            continue
                
                return mac
            else:
                print("Invalid device number")
        except ValueError:
            print("Invalid input")


def list_serial_ports():
    """List available serial ports (for pyomyo/dongle)."""
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        
        print("\n=== Serial Ports (for dongle) ===\n")
        for port in ports:
            info = f"  {port.device}"
            if port.description:
                info += f" - {port.description}"
            print(info)
        
        if not ports:
            print("  No serial ports found")
        print()
        return [p.device for p in ports]
    except ImportError:
        print("pyserial not installed")
        return []


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Stream Myo EMG and IMU data to LSL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Backends:
  dl-myo  - Native Bluetooth (no dongle needed) - RECOMMENDED
  pyomyo  - Requires the Myo Bluetooth dongle (EMG only, no IMU)
  mock    - Fake data for testing

Streams created:
  {name}_EMG  - 8 channels @ 200Hz (EMG from 8 pod sensors)
  {name}_IMU  - 10 channels @ 50Hz (orientation + accel + gyro)

Examples:
  # RECOMMENDED: Interactive device selection
  python myo_interface.py --select
  
  # Scan for Myo devices (shows MAC addresses)
  python myo_interface.py --scan
  
  # Ping a specific Myo (it will vibrate twice)
  python myo_interface.py --ping D2:3B:85:94:32:8E
  
  # Connect to specific Myo by MAC address
  python myo_interface.py --mac D2:3B:85:94:32:8E
  
  # Auto-detect (connects to first Myo found)
  python myo_interface.py
  
  # Use mock data for testing (no hardware needed)
  python myo_interface.py --mock

Multiple Myos in classroom:
  1. Run: python myo_interface.py --select
  2. All nearby Myos will be listed with numbers
  3. Ping your Myo to identify it (it vibrates)
  4. Enter the number to select and start streaming
  
IMPORTANT: Close MyoConnect before running!
        """
    )
    
    # Device selection options
    parser.add_argument("--select", action="store_true",
                       help="Interactive mode: scan, ping, and select a Myo")
    parser.add_argument("--scan", action="store_true",
                       help="Scan for Myo devices and exit")
    parser.add_argument("--ping",
                       help="Ping a Myo by MAC address (it will vibrate)")
    parser.add_argument("--mac",
                       help="MAC address of Myo to connect to")
    
    # Backend options
    parser.add_argument("--backend", default="auto",
                       choices=["auto", "dl-myo", "pyomyo"],
                       help="Myo backend: dl-myo (native BT), pyomyo (dongle), auto")
    parser.add_argument("--mock", action="store_true",
                       help="Use mock data (no hardware)")
    parser.add_argument("--tty",
                       help="Serial port for dongle (pyomyo only)")
    parser.add_argument("--list-ports", action="store_true",
                       help="List serial ports (for pyomyo)")
    
    # Stream options
    parser.add_argument("--stream", default="Myo",
                       help="LSL stream name prefix (default: Myo)")
    parser.add_argument("--mode", default="raw",
                       choices=["raw", "filtered", "preprocessed"],
                       help="EMG mode (default: raw)")
    parser.add_argument("--no-imu", action="store_true",
                       help="Disable IMU streaming (EMG only)")
    parser.add_argument("--duration", type=int, default=0,
                       help="Duration in seconds (0 = run until Ctrl+C)")
    
    args = parser.parse_args()
    
    # Handle scan command
    if args.scan:
        asyncio.run(scan_for_myos())
        return 0
    
    # Handle ping command
    if args.ping:
        success = asyncio.run(ping_myo(args.ping))
        return 0 if success else 1
    
    # Handle list-ports command
    if args.list_ports:
        list_serial_ports()
        return 0
    
    # Handle interactive select mode
    selected_mac = None
    if args.select:
        selected_mac = asyncio.run(interactive_select())
        if selected_mac is None:
            print("\nNo device selected. Exiting.")
            return 0
        print(f"\nStarting stream with MAC: {selected_mac}\n")
    
    enable_imu = not args.no_imu
    
    # Determine MAC address to use
    mac_to_use = selected_mac or args.mac
    
    # Create streamer
    if args.mock:
        streamer = MockMyoStreamer(stream_name=args.stream, enable_imu=enable_imu)
    elif args.backend == "pyomyo" or (args.tty and not mac_to_use):
        streamer = create_streamer("pyomyo", stream_name=args.stream, 
                                   mode=args.mode, tty=args.tty)
    elif mac_to_use:
        streamer = create_streamer("dl-myo", stream_name=args.stream,
                                   mode=args.mode, mac=mac_to_use, enable_imu=enable_imu)
    else:
        streamer = create_streamer(args.backend, stream_name=args.stream,
                                   mode=args.mode, enable_imu=enable_imu)
    
    # Run
    try:
        streamer.start()
        
        if args.duration > 0:
            print(f"\nStreaming for {args.duration} seconds...")
            for i in range(args.duration):
                time.sleep(1)
                status = f"  {i+1}/{args.duration}s - EMG: {streamer.sample_count}"
                if enable_imu and hasattr(streamer, 'imu_sample_count'):
                    status += f" | IMU: {streamer.imu_sample_count}"
                print(status, end='\r')
            print()
        else:
            print("\nStreaming... Press Ctrl+C to stop\n")
            while True:
                time.sleep(1)
                status = f"  EMG: {streamer.sample_count}"
                if enable_imu and hasattr(streamer, 'imu_sample_count'):
                    status += f" | IMU: {streamer.imu_sample_count}"
                print(status + "    ", end='\r')
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        streamer.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())