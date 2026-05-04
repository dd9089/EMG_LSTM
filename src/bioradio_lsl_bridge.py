#!/usr/bin/env python3
"""
BioRadio LSL Network Bridge
==============================

Streams BioRadio data from a Windows machine to any other machine (Mac/Linux)
over the network using Lab Streaming Layer (LSL).

Architecture:
    [BioRadio] --BT--> [Windows PC] --LSL/network--> [Mac/Linux]

Two components:
    1. SENDER (runs on Windows, where BioRadio connects via Bluetooth):
       - Connects to BioRadio via serial port
       - Reads data and pushes to LSL outlet
       - Run: python bioradio_lsl_bridge.py --send --port COM9

    2. RECEIVER (runs on Mac or any machine on the same network):
       - Discovers the LSL stream on the network
       - Provides a BioRadio-compatible interface for lab scripts
       - Run: python bioradio_lsl_bridge.py --receive

Both machines must be on the same network (LSL uses multicast UDP for
discovery and TCP for data transfer).

Requirements:
    pip install pylsl pyserial

Usage:
    # On Windows (where BioRadio is paired):
    python bioradio_lsl_bridge.py --send --port COM9

    # On Mac (in the lab):
    python bioradio_lsl_bridge.py --receive

    # Or use the BioRadioLSL class in your lab scripts:
    from bioradio_lsl_bridge import BioRadioLSL
    radio = BioRadioLSL()
    radio.connect()  # finds the LSL stream on the network
    data = radio.read_data()
"""

import sys
import os
import time
import argparse
import logging
import struct
import threading
from collections import deque
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LSL_STREAM_NAME = "BioRadio"
LSL_STREAM_TYPE = "EEG"  # Standard type for physiological data
LSL_SOURCE_ID = "BioRadio_Bridge"

SYNC_BYTE = 0xF0
CMD_GET_FIRMWARE = bytes([0xF0, 0xF1, 0x00])
CMD_GET_MODE = bytes([0xF0, 0x30])
CMD_SET_IDLE = bytes([0xF0, 0x21, 0x00])
CMD_START_ACQ = bytes([0xF0, 0x21, 0x01])
CMD_STOP_ACQ = bytes([0xF0, 0x21, 0x00])


# ---------------------------------------------------------------------------
# SENDER: Windows side — reads BioRadio, pushes to LSL
# ---------------------------------------------------------------------------
class BioRadioLSLSender:
    """
    Runs on the Windows machine where BioRadio is connected via Bluetooth.
    Reads raw data from the serial port and pushes it to an LSL outlet.

    Can operate in two modes:
      - RAW mode: streams raw serial bytes (receiver must parse BioRadio protocol)
      - PARSED mode: uses bioradio.py to parse data, streams individual channels

    RAW mode is simpler and more reliable; PARSED mode requires bioradio.py on
    the Windows side but gives cleaner data on the receiver.
    """

    def __init__(self, port: str, baud: int = 460800, mode: str = "raw"):
        self.port = port
        self.baud = baud
        self.mode = mode
        self._running = False
        self._outlet = None
        self._serial = None

    def start(self):
        """Start streaming BioRadio data to LSL."""
        try:
            import pylsl
        except ImportError:
            print("ERROR: pylsl not installed. Install with: pip install pylsl")
            sys.exit(1)

        try:
            import serial
        except ImportError:
            print("ERROR: pyserial not installed. Install with: pip install pyserial")
            sys.exit(1)

        if self.mode == "raw":
            self._start_raw(pylsl, serial)
        else:
            self._start_parsed(pylsl, serial)

    def _start_raw(self, pylsl, serial_module):
        """Stream raw serial bytes over LSL."""
        print(f"Opening serial port {self.port} at {self.baud} baud...")
        self._serial = serial_module.Serial(
            port=self.port,
            baudrate=self.baud,
            timeout=1.0,
            write_timeout=1.0,
        )
        time.sleep(0.5)

        # Create LSL outlet for raw bytes
        # Use irregular rate since we're streaming raw chunks
        stream_info = pylsl.StreamInfo(
            name=LSL_STREAM_NAME,
            type="Raw",
            channel_count=1,
            nominal_srate=pylsl.IRREGULAR_RATE,
            channel_format=pylsl.cf_string,
            source_id=LSL_SOURCE_ID + "_raw",
        )

        # Add description
        desc = stream_info.desc()
        desc.append_child_value("manufacturer", "GLNeuroTech")
        desc.append_child_value("model", "BioRadio")
        desc.append_child_value("serial_port", self.port)
        desc.append_child_value("baud_rate", str(self.baud))
        desc.append_child_value("mode", "raw")

        self._outlet = pylsl.StreamOutlet(stream_info)

        print(f"\n{'='*60}")
        print(f"  BioRadio LSL Bridge — SENDER (Raw Mode)")
        print(f"  Port: {self.port}")
        print(f"  Stream: '{LSL_STREAM_NAME}' (type: Raw)")
        print(f"  The receiver can now discover this stream on the network")
        print(f"{'='*60}\n")

        self._running = True
        bytes_sent = 0
        chunks_sent = 0

        try:
            while self._running:
                # Read available data from serial
                if self._serial.in_waiting > 0:
                    data = self._serial.read(self._serial.in_waiting)
                    if data:
                        # Send as hex string (LSL string channel)
                        hex_str = data.hex()
                        self._outlet.push_sample([hex_str])
                        bytes_sent += len(data)
                        chunks_sent += 1

                        if chunks_sent % 100 == 0:
                            print(f"  Sent {chunks_sent} chunks ({bytes_sent} bytes)")
                else:
                    time.sleep(0.001)  # Small sleep to avoid busy-wait

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self._running = False
            if self._serial:
                self._serial.close()
            print(f"Total: {chunks_sent} chunks, {bytes_sent} bytes")

    def _start_parsed(self, pylsl, serial_module):
        """Stream parsed BioRadio channels over LSL."""
        # Import bioradio module
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            from bioradio import BioRadio
        except ImportError:
            print("ERROR: bioradio.py not found. Place it in the same directory.")
            print("Falling back to raw mode...")
            self._start_raw(pylsl, serial_module)
            return

        print(f"Connecting to BioRadio on {self.port}...")
        radio = BioRadio(port=self.port)
        radio.connect()

        config = radio.get_configuration()
        n_channels = len(config.enabled_channels) if hasattr(config, 'enabled_channels') else 8
        sample_rate = config.sample_rate if hasattr(config, 'sample_rate') else 250

        # Create LSL outlet for parsed channels
        stream_info = pylsl.StreamInfo(
            name=LSL_STREAM_NAME,
            type=LSL_STREAM_TYPE,
            channel_count=n_channels,
            nominal_srate=sample_rate,
            channel_format=pylsl.cf_float32,
            source_id=LSL_SOURCE_ID + "_parsed",
        )

        # Add channel labels
        desc = stream_info.desc()
        desc.append_child_value("manufacturer", "GLNeuroTech")
        desc.append_child_value("model", "BioRadio")
        channels = desc.append_child("channels")
        for i in range(n_channels):
            ch = channels.append_child("channel")
            ch.append_child_value("label", f"Ch{i+1}")
            ch.append_child_value("unit", "microvolts")
            ch.append_child_value("type", "EEG")

        self._outlet = pylsl.StreamOutlet(stream_info)

        print(f"\n{'='*60}")
        print(f"  BioRadio LSL Bridge — SENDER (Parsed Mode)")
        print(f"  Port: {self.port}")
        print(f"  Channels: {n_channels} @ {sample_rate} Hz")
        print(f"  Stream: '{LSL_STREAM_NAME}' (type: {LSL_STREAM_TYPE})")
        print(f"{'='*60}\n")

        # Start acquisition
        radio.start_acquisition()
        self._running = True
        samples_sent = 0

        try:
            while self._running:
                data = radio.read_data(timeout=1.0)
                if data and hasattr(data, 'samples'):
                    for sample in data.samples:
                        self._outlet.push_sample(sample)
                        samples_sent += 1

                    if samples_sent % 1000 == 0:
                        print(f"  Sent {samples_sent} samples")
                else:
                    time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self._running = False
            radio.stop_acquisition()
            radio.disconnect()
            print(f"Total: {samples_sent} samples")


# ---------------------------------------------------------------------------
# RECEIVER: Mac side — discovers LSL stream, provides BioRadio-like API
# ---------------------------------------------------------------------------
class BioRadioLSL:
    """
    Receiver that discovers a BioRadio LSL stream on the network and provides
    a simplified API for reading data. Use this on the Mac side.

    Usage:
        radio = BioRadioLSL()
        radio.connect()       # blocks until the LSL stream is found
        radio.start_acquisition()

        while True:
            samples = radio.read_samples(timeout=1.0)
            if samples:
                for sample, timestamp in samples:
                    print(f"t={timestamp:.3f}: {sample}")

        radio.stop_acquisition()
        radio.disconnect()
    """

    def __init__(self, stream_name: str = LSL_STREAM_NAME, timeout: float = 30.0):
        self._stream_name = stream_name
        self._timeout = timeout
        self._inlet = None
        self._stream_info = None
        self._is_connected = False
        self._is_acquiring = False
        self._buffer = deque(maxlen=10000)
        self._reader_thread = None
        self._raw_mode = False

    @property
    def is_connected(self):
        return self._is_connected

    @property
    def channel_count(self):
        if self._stream_info:
            return self._stream_info.channel_count()
        return 0

    @property
    def sample_rate(self):
        if self._stream_info:
            return self._stream_info.nominal_srate()
        return 0

    def connect(self):
        """Discover and connect to the BioRadio LSL stream."""
        try:
            import pylsl
        except ImportError:
            print("ERROR: pylsl not installed. Install with: pip install pylsl")
            raise ImportError("pylsl required: pip install pylsl")

        print(f"Searching for LSL stream '{self._stream_name}' on network...")
        print(f"(Make sure the sender is running on the Windows machine)")

        streams = pylsl.resolve_byprop(
            "name", self._stream_name, timeout=self._timeout
        )

        if not streams:
            raise ConnectionError(
                f"No LSL stream named '{self._stream_name}' found on the network.\n"
                f"Make sure the sender (bioradio_lsl_bridge.py --send) is running "
                f"on the Windows machine and both machines are on the same network."
            )

        self._stream_info = streams[0]
        stream_type = self._stream_info.type()
        self._raw_mode = stream_type == "Raw"

        print(f"Found stream: {self._stream_info.name()} "
              f"(type={stream_type}, "
              f"channels={self._stream_info.channel_count()}, "
              f"rate={self._stream_info.nominal_srate()})")

        # Open inlet
        self._inlet = pylsl.StreamInlet(self._stream_info)
        self._is_connected = True
        print("Connected to BioRadio LSL stream!")

    def start_acquisition(self):
        """Start reading data from the LSL stream in a background thread."""
        if not self._is_connected:
            raise RuntimeError("Not connected — call connect() first")

        self._is_acquiring = True
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="LSL-Reader"
        )
        self._reader_thread.start()
        print("Acquisition started")

    def _reader_loop(self):
        """Background thread that reads from LSL and buffers data."""
        while self._is_acquiring and self._inlet:
            try:
                if self._raw_mode:
                    sample, timestamp = self._inlet.pull_sample(timeout=0.1)
                    if sample:
                        # Decode hex string back to bytes
                        raw_bytes = bytes.fromhex(sample[0])
                        self._buffer.append((raw_bytes, timestamp))
                else:
                    sample, timestamp = self._inlet.pull_sample(timeout=0.1)
                    if sample:
                        self._buffer.append((sample, timestamp))
            except Exception as e:
                logger.debug(f"LSL read error: {e}")
                time.sleep(0.01)

    def read_samples(self, max_samples: int = 100, timeout: float = 1.0):
        """
        Read available samples from the buffer.

        Returns: list of (sample, timestamp) tuples
        """
        if not self._is_acquiring:
            return []

        deadline = time.monotonic() + timeout
        result = []

        while len(result) < max_samples:
            if self._buffer:
                result.append(self._buffer.popleft())
            elif time.monotonic() >= deadline:
                break
            else:
                time.sleep(0.01)

        return result

    def read_raw_bytes(self, timeout: float = 1.0) -> bytes:
        """
        Read raw bytes from the stream (raw mode only).
        Returns concatenated bytes from all available chunks.
        """
        if not self._raw_mode:
            raise RuntimeError("read_raw_bytes() only works in raw mode")

        samples = self.read_samples(timeout=timeout)
        return b"".join(raw for raw, ts in samples)

    def stop_acquisition(self):
        """Stop reading data."""
        self._is_acquiring = False
        if self._reader_thread:
            self._reader_thread.join(timeout=3.0)
            self._reader_thread = None
        print("Acquisition stopped")

    def disconnect(self):
        """Disconnect from the LSL stream."""
        self.stop_acquisition()
        self._inlet = None
        self._is_connected = False
        print("Disconnected from LSL stream")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BioRadio LSL Network Bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # On Windows (sender):
  python bioradio_lsl_bridge.py --send --port COM9

  # On Windows (sender, parsed mode — requires bioradio.py):
  python bioradio_lsl_bridge.py --send --port COM9 --mode parsed

  # On Mac (receiver — displays incoming data):
  python bioradio_lsl_bridge.py --receive

  # On Mac (in your own script):
  from bioradio_lsl_bridge import BioRadioLSL
  radio = BioRadioLSL()
  radio.connect()
  radio.start_acquisition()
  samples = radio.read_samples()
"""
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--send", action="store_true",
                       help="Run as SENDER (on Windows, reads BioRadio serial)")
    group.add_argument("--receive", action="store_true",
                       help="Run as RECEIVER (on Mac, reads LSL stream)")

    parser.add_argument("--port", "-p", default=None,
                        help="Serial port (sender mode, e.g. COM9)")
    parser.add_argument("--baud", type=int, default=460800,
                        help="Baud rate (default: 460800)")
    parser.add_argument("--mode", choices=["raw", "parsed"], default="raw",
                        help="Sender mode: 'raw' (bytes) or 'parsed' (channels)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Receiver: timeout for stream discovery (default: 30s)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s [%(levelname)s] %(message)s")
    else:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")

    if args.send:
        if not args.port:
            print("ERROR: --port is required for sender mode")
            print("  Example: python bioradio_lsl_bridge.py --send --port COM9")
            sys.exit(1)

        sender = BioRadioLSLSender(port=args.port, baud=args.baud, mode=args.mode)
        sender.start()

    elif args.receive:
        receiver = BioRadioLSL(timeout=args.timeout)

        print(f"{'='*60}")
        print(f"  BioRadio LSL Bridge — RECEIVER")
        print(f"  Searching for LSL stream on network...")
        print(f"{'='*60}\n")

        try:
            receiver.connect()
        except ConnectionError as e:
            print(f"\nERROR: {e}")
            sys.exit(1)

        receiver.start_acquisition()

        print("\nReceiving data (Ctrl+C to stop)...\n")

        samples_received = 0
        start_time = time.monotonic()

        try:
            while True:
                samples = receiver.read_samples(max_samples=100, timeout=0.5)
                for sample, timestamp in samples:
                    samples_received += 1
                    if receiver._raw_mode:
                        print(f"  [{samples_received:6d}] t={timestamp:.3f} "
                              f"raw={sample.hex(' ') if isinstance(sample, bytes) else sample}")
                    else:
                        # Show first few channels
                        channels = sample[:4]
                        more = f" ... +{len(sample)-4}" if len(sample) > 4 else ""
                        print(f"  [{samples_received:6d}] t={timestamp:.3f} "
                              f"ch={[f'{v:.2f}' for v in channels]}{more}")

                    if samples_received % 500 == 0:
                        elapsed = time.monotonic() - start_time
                        rate = samples_received / elapsed if elapsed > 0 else 0
                        print(f"\n  --- {samples_received} samples in {elapsed:.1f}s "
                              f"({rate:.1f} samples/s) ---\n")

        except KeyboardInterrupt:
            elapsed = time.monotonic() - start_time
            print(f"\n\nTotal: {samples_received} samples in {elapsed:.1f}s")

        receiver.disconnect()


if __name__ == "__main__":
    main()
