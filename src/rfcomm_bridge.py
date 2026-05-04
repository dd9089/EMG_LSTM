#!/usr/bin/env python3
"""
RFCOMM Bridge for BioRadio on macOS
=====================================

Bypasses the broken /dev/cu.* serial port by opening an RFCOMM channel
directly via macOS IOBluetooth framework. Provides a serial.Serial-compatible
interface so bioradio.py can use it as a drop-in replacement.

The problem: macOS sometimes pairs Bluetooth SPP devices with the wrong
service profile (e.g., "Braille ACL" instead of Serial Port Profile).
The /dev/cu.* device node exists but no RFCOMM transport is behind it,
so pyserial can open the port but never receives any data.

This module uses pyobjc + IOBluetooth to open the RFCOMM channel directly,
bypassing the macOS Bluetooth serial port driver entirely.

Requirements:
    pip install pyobjc-framework-IOBluetooth

Usage:
    from rfcomm_bridge import RFCOMMSerialBridge

    # Use exactly like serial.Serial:
    ser = RFCOMMSerialBridge("EC:FE:7E:12:BA:36", channel_id=1)
    ser.open()
    ser.write(b"\\xf0\\xf1\\x00")
    data = ser.read(10)
    ser.close()
"""

import sys
import time
import threading
import logging
from collections import deque
from typing import Optional

if sys.platform != "darwin":
    raise ImportError("rfcomm_bridge is macOS-only (requires IOBluetooth framework)")

try:
    import objc
    import Foundation
    import IOBluetooth
except ImportError as e:
    raise ImportError(
        "pyobjc-framework-IOBluetooth is required.\n"
        "Install with: pip install pyobjc-framework-IOBluetooth"
    ) from e

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RFCOMM Channel Delegate
# ---------------------------------------------------------------------------
class _RFCOMMDelegate(Foundation.NSObject):
    """
    Objective-C delegate that receives callbacks from IOBluetoothRFCOMMChannel.
    Data arrives asynchronously and is buffered in a thread-safe deque.
    """

    def init(self):
        self = objc.super(_RFCOMMDelegate, self).init()
        if self is None:
            return None
        self._buffer = deque()
        self._buffer_lock = threading.Lock()
        self._data_event = threading.Event()
        self._is_open = False
        self._open_event = threading.Event()
        self._open_error = None
        self._closed_event = threading.Event()
        return self

    # --- IOBluetoothRFCOMMChannel delegate methods ---

    def rfcommChannelOpenComplete_status_(self, channel, status):
        """Called when the RFCOMM channel open completes."""
        if status == 0:  # kIOReturnSuccess
            logger.info("RFCOMM channel opened successfully")
            self._is_open = True
        else:
            logger.error(f"RFCOMM channel open failed with status: {status}")
            self._open_error = status
        self._open_event.set()

    def rfcommChannelData_data_length_(self, channel, data, length):
        """Called when data arrives on the RFCOMM channel."""
        # data is a raw pointer — convert to bytes
        raw_bytes = bytes(data[:length])
        with self._buffer_lock:
            self._buffer.extend(raw_bytes)
        self._data_event.set()
        logger.debug(f"RFCOMM RX: {len(raw_bytes)} bytes")

    def rfcommChannelClosed_(self, channel):
        """Called when the RFCOMM channel is closed."""
        logger.info("RFCOMM channel closed")
        self._is_open = False
        self._closed_event.set()
        self._data_event.set()  # Wake up any blocking reads

    # --- Buffer access methods (called from Python threads) ---

    @property
    def in_waiting(self) -> int:
        with self._buffer_lock:
            return len(self._buffer)

    def read_bytes(self, count: int) -> bytes:
        """Read up to count bytes from the buffer."""
        with self._buffer_lock:
            available = min(count, len(self._buffer))
            if available == 0:
                return b""
            result = bytes([self._buffer.popleft() for _ in range(available)])
            if not self._buffer:
                self._data_event.clear()
            return result

    def clear_buffer(self):
        """Discard all buffered data."""
        with self._buffer_lock:
            self._buffer.clear()
            self._data_event.clear()


# ---------------------------------------------------------------------------
# RunLoop Thread
# ---------------------------------------------------------------------------
class _RunLoopThread(threading.Thread):
    """
    Runs the Cocoa NSRunLoop on a dedicated thread.

    IOBluetooth dispatches delegate callbacks on the thread that initiated
    the connection. So we must both:
      1. Open the RFCOMM channel FROM this thread
      2. Run the NSRunLoop ON this thread to receive callbacks

    Use schedule_and_wait() to execute callables on this thread.
    """

    def __init__(self):
        super().__init__(daemon=True, name="RFCOMM-RunLoop")
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._run_loop = None
        # Queue for work items to execute on the run loop thread
        self._work_queue = deque()
        self._work_lock = threading.Lock()

    def run(self):
        self._run_loop = Foundation.NSRunLoop.currentRunLoop()
        self._started_event.set()
        logger.debug("RunLoop thread started")

        while not self._stop_event.is_set():
            pool = Foundation.NSAutoreleasePool.alloc().init()

            # Process any queued work items
            with self._work_lock:
                while self._work_queue:
                    work_item = self._work_queue.popleft()
                    try:
                        func, result_holder, done_event = work_item
                        result_holder["result"] = func()
                    except Exception as e:
                        result_holder["error"] = e
                    finally:
                        done_event.set()

            # Run the loop to process IOBluetooth callbacks
            self._run_loop.runMode_beforeDate_(
                Foundation.NSDefaultRunLoopMode,
                Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.05)
            )
            del pool

        logger.debug("RunLoop thread stopped")

    def schedule_and_wait(self, func, timeout=30.0):
        """
        Schedule a callable to run on the RunLoop thread and wait for it.
        Returns the callable's return value, or raises its exception.
        """
        result_holder = {"result": None, "error": None}
        done_event = threading.Event()
        with self._work_lock:
            self._work_queue.append((func, result_holder, done_event))
        if not done_event.wait(timeout):
            raise TimeoutError("Timed out waiting for RunLoop thread to execute work")
        if result_holder["error"] is not None:
            raise result_holder["error"]
        return result_holder["result"]

    def stop(self):
        self._stop_event.set()

    def wait_started(self, timeout=5.0):
        return self._started_event.wait(timeout)


# ---------------------------------------------------------------------------
# RFCOMMSerialBridge — serial.Serial-compatible wrapper
# ---------------------------------------------------------------------------
class RFCOMMSerialBridge:
    """
    Drop-in replacement for serial.Serial that uses IOBluetooth RFCOMM.

    Provides the subset of the serial.Serial API used by bioradio.py:
        - Properties: is_open, in_waiting, timeout, dtr, rts
        - Methods: read(n), write(data), flush(), close()

    Parameters:
        address: Bluetooth MAC address (e.g., "EC:FE:7E:12:BA:36")
        channel_id: RFCOMM channel ID (default 1, which is standard for SPP)
        timeout: Read timeout in seconds (default 2.0)
        write_timeout: Write timeout in seconds (default 2.0)
    """

    def __init__(self, address: str, channel_id: int = 1,
                 baudrate: int = 460800,
                 timeout: float = 2.0, write_timeout: float = 2.0,
                 rtscts: bool = False, dsrdtr: bool = False,
                 **kwargs):
        self._address = address
        self._channel_id = channel_id
        self._baudrate = baudrate  # Stored but not used — RFCOMM has no baud rate
        self._timeout = timeout
        self._write_timeout = write_timeout
        self._channel = None
        self._delegate = None
        self._runloop_thread = None
        self._device = None
        self._is_open = False

        # These are no-ops for RFCOMM but bioradio.py sets them
        self._dtr = False
        self._rts = False

    # --- Properties matching serial.Serial ---

    @property
    def is_open(self) -> bool:
        return self._is_open and self._delegate is not None and self._delegate._is_open

    @property
    def in_waiting(self) -> int:
        if self._delegate is None:
            return 0
        return self._delegate.in_waiting

    @property
    def timeout(self) -> float:
        return self._timeout

    @timeout.setter
    def timeout(self, value: float):
        self._timeout = value

    @property
    def dtr(self) -> bool:
        return self._dtr

    @dtr.setter
    def dtr(self, value: bool):
        self._dtr = value  # No-op for RFCOMM

    @property
    def rts(self) -> bool:
        return self._rts

    @rts.setter
    def rts(self, value: bool):
        self._rts = value  # No-op for RFCOMM

    # --- SDP Discovery ---

    def _perform_sdp_query(self) -> list:
        """
        Perform a fresh SDP query on the device and return a list of
        (channel_id, service_name) tuples for all RFCOMM-based services.
        """
        discovered = []

        logger.info("Performing SDP service discovery ...")

        # SPP UUID: 00001101-0000-1000-8000-00805F9B34FB
        spp_uuid = IOBluetooth.IOBluetoothSDPUUID.uuid16_(0x1101)

        # Perform a fresh SDP query (clears cached results)
        result = self._device.performSDPQuery_(None)
        if result != 0:
            logger.warning(f"SDP query returned error {result}, trying cached services")

        # Give SDP query time to complete
        time.sleep(2.0)

        # Get all service records from the device
        services = self._device.services()
        if services is None or len(services) == 0:
            logger.warning("No SDP service records found on device")
            return discovered

        for svc in services:
            svc_name = "unknown"
            try:
                # Get service name
                name_attr = svc.attributeDataElement()
                svc_name = str(svc.getServiceName()) if svc.getServiceName() else "unnamed"
            except Exception:
                pass

            # Check if this service has an RFCOMM channel
            try:
                # getRFCOMMChannelID_ returns (result, channel_id)
                res, ch_id = svc.getRFCOMMChannelID_(None)
                if res == 0:
                    discovered.append((ch_id, svc_name))
                    logger.info(f"  SDP service: '{svc_name}' -> RFCOMM channel {ch_id}")
            except TypeError:
                try:
                    res, ch_id = svc.getRFCOMMChannelID_()
                    if res == 0:
                        discovered.append((ch_id, svc_name))
                        logger.info(f"  SDP service: '{svc_name}' -> RFCOMM channel {ch_id}")
                except Exception as e:
                    logger.debug(f"  SDP service '{svc_name}': no RFCOMM channel ({e})")
            except Exception as e:
                logger.debug(f"  SDP service '{svc_name}': error getting RFCOMM channel ({e})")

            # Also check if this service matches the SPP UUID
            try:
                if svc.hasServiceFromArray_([spp_uuid]):
                    logger.info(f"  -> Service '{svc_name}' matches SPP UUID (0x1101)")
            except Exception:
                pass

        if not discovered:
            logger.warning("No RFCOMM services found via SDP")
            # Log all services for debugging
            for svc in services:
                try:
                    svc_name = str(svc.getServiceName()) if svc.getServiceName() else "unnamed"
                    logger.info(f"  SDP service (non-RFCOMM): '{svc_name}'")
                    # Try to get the service class UUIDs
                    try:
                        attrs = svc.attributes()
                        if attrs:
                            for key in attrs:
                                logger.debug(f"    attr {key}: {attrs[key]}")
                    except Exception:
                        pass
                except Exception:
                    pass

        return discovered

    def _try_open_rfcomm(self, channel_id: int, use_async: bool = False) -> int:
        """
        Try to open a specific RFCOMM channel. Runs on the RunLoop thread
        so that delegate callbacks are dispatched correctly.

        Returns IOKit result code (0 = success).
        """
        delegate = self._delegate
        device = self._device

        def _do_open():
            if use_async:
                method = device.openRFCOMMChannelAsync_withChannelID_delegate_
            else:
                method = device.openRFCOMMChannelSync_withChannelID_delegate_
            try:
                result, channel = method(None, channel_id, delegate)
            except TypeError:
                # Older PyObjC may not pass the output parameter
                result, channel = method(channel_id, delegate)
            return (result, channel)

        try:
            result, channel = self._runloop_thread.schedule_and_wait(_do_open, timeout=10.0)
            self._channel = channel
            return result
        except Exception as e:
            logger.debug(f"  RFCOMM open ch={channel_id} {'async' if use_async else 'sync'} error: {e}")
            return -1

    # --- Connection ---

    def open(self, connect_timeout: float = 15.0) -> bool:
        """
        Open the RFCOMM channel to the BioRadio.

        Performs SDP discovery to find the correct RFCOMM channel,
        then opens it. If no channel ID is in the SDP record, brute-forces
        channels 1-30. Returns True on success, raises ConnectionError
        on failure.
        """
        if self._is_open:
            return True

        logger.info(f"Opening RFCOMM connection to {self._address}")

        # Start the run loop thread — ALL IOBluetooth calls must happen on this
        # thread so delegate callbacks are dispatched correctly
        self._runloop_thread = _RunLoopThread()
        self._runloop_thread.start()
        if not self._runloop_thread.wait_started(5.0):
            raise ConnectionError("Failed to start RunLoop thread")

        # Create the delegate (must be retained for the lifetime of the connection)
        self._delegate = _RFCOMMDelegate.alloc().init()

        # Get the Bluetooth device by address — this is safe from any thread
        self._device = IOBluetooth.IOBluetoothDevice.withAddressString_(self._address)
        if self._device is None:
            raise ConnectionError(
                f"Could not find Bluetooth device with address {self._address}. "
                f"Make sure the BioRadio is paired in System Settings > Bluetooth."
            )

        logger.info(f"Found device: {self._device.name()} ({self._address})")

        # Open baseband connection on the RunLoop thread
        if not self._device.isConnected():
            logger.info("Opening baseband connection ...")
            def _open_baseband():
                return self._device.openConnection()
            result = self._runloop_thread.schedule_and_wait(_open_baseband, timeout=10.0)
            if result != 0:
                raise ConnectionError(
                    f"Failed to open baseband connection (error {result}). "
                    f"Make sure the BioRadio is powered on and in range."
                )
            time.sleep(1.0)

        # Step 1: SDP discovery on the RunLoop thread
        def _sdp_query():
            return self._device.performSDPQuery_(None)
        logger.info("Performing SDP query ...")
        sdp_result = self._runloop_thread.schedule_and_wait(_sdp_query, timeout=10.0)
        logger.info(f"SDP query result: {sdp_result}")
        time.sleep(2.0)

        # Check SDP services for RFCOMM channel
        sdp_channels = self._perform_sdp_query()

        # Build channel list: SDP-discovered first, then user-specified, then 1-30
        channels_to_try = []
        for ch_id, svc_name in sdp_channels:
            channels_to_try.append(ch_id)
        if self._channel_id not in channels_to_try:
            channels_to_try.append(self._channel_id)
        for ch in range(1, 31):
            if ch not in channels_to_try:
                channels_to_try.append(ch)

        logger.info(f"Will try {len(channels_to_try)} channels")

        # Step 2: Try each channel — both sync and async, all on RunLoop thread
        result = -1
        for ch_id in channels_to_try:
            # Try synchronous open
            self._delegate = _RFCOMMDelegate.alloc().init()
            result = self._try_open_rfcomm(ch_id, use_async=False)
            if result == 0:
                self._channel_id = ch_id
                logger.info(f"  Channel {ch_id} (sync): opened, waiting for delegate ...")
                # For sync, the channel should be open immediately
                # but wait a bit for the delegate callback
                self._delegate._open_event.wait(timeout=3.0)
                if self._delegate._is_open:
                    logger.info(f"  Channel {ch_id} (sync): CONNECTED")
                    break
                else:
                    logger.info(f"  Channel {ch_id} (sync): returned 0 but delegate not confirmed")
                    # Still might work — the sync call opened it
                    self._delegate._is_open = True
                    break

            # Try asynchronous open
            self._delegate = _RFCOMMDelegate.alloc().init()
            result = self._try_open_rfcomm(ch_id, use_async=True)
            if result == 0:
                logger.info(f"  Channel {ch_id} (async): initiated, waiting for callback ...")
                # Wait for the async callback with longer timeout
                self._delegate._open_event.wait(timeout=5.0)
                if self._delegate._is_open:
                    self._channel_id = ch_id
                    logger.info(f"  Channel {ch_id} (async): CONNECTED")
                    break
                elif self._delegate._open_error is not None:
                    logger.debug(f"  Channel {ch_id} (async): failed with {self._delegate._open_error}")
                    result = self._delegate._open_error
                else:
                    logger.debug(f"  Channel {ch_id} (async): no callback received")
                    result = -1
            else:
                logger.debug(f"  Channel {ch_id}: failed (error {result})")

        if result != 0 and not self._delegate._is_open:
            raise ConnectionError(
                f"Failed to open RFCOMM channel (last error {result}).\n"
                f"SDP discovered channels: {sdp_channels}\n"
                f"Tried channels: 1-30 (sync + async on RunLoop thread)\n\n"
                f"The BioRadio advertises SPP UUID (0x1101) but the RFCOMM\n"
                f"connection cannot be established from macOS.\n\n"
                f"Possible workarounds:\n"
                f"  1. Try: sudo pkill bluetoothd  (restart BT daemon)\n"
                f"     Then power-cycle the BioRadio and re-pair\n"
                f"  2. Delete BT plist: sudo rm /Library/Preferences/com.apple.Bluetooth.plist\n"
                f"     Then reboot and re-pair\n"
                f"  3. Check if a firmware update is available for the BioRadio"
            )

        # Set the delegate on the channel
        self._channel.setDelegate_(self._delegate)

        # Wait for the open to complete
        if not self._delegate._open_event.wait(timeout=connect_timeout):
            raise ConnectionError(
                f"RFCOMM channel open timed out after {connect_timeout}s"
            )

        if self._delegate._open_error is not None:
            raise ConnectionError(
                f"RFCOMM channel open failed with error: {self._delegate._open_error}"
            )

        self._is_open = True
        logger.info(f"RFCOMM channel {self._channel_id} open to {self._device.name()}")
        return True

    # --- serial.Serial-compatible I/O methods ---

    def read(self, size: int = 1) -> bytes:
        """
        Read up to `size` bytes. Blocks up to self.timeout seconds.
        Returns available bytes (may be fewer than size), or b"" on timeout.
        """
        if not self.is_open:
            return b""

        deadline = time.monotonic() + (self._timeout or 0)
        result = bytearray()

        while len(result) < size:
            # Check buffer
            available = self._delegate.read_bytes(size - len(result))
            if available:
                result.extend(available)
                if len(result) >= size:
                    break

            # If we have some data and timeout is set, return what we have
            if result and self._timeout is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break

            # Wait for more data
            if self._timeout is None:
                # Blocking forever
                self._delegate._data_event.wait(0.1)
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._delegate._data_event.wait(min(0.1, remaining))

        return bytes(result)

    def write(self, data: bytes) -> int:
        """Write data to the RFCOMM channel."""
        if not self.is_open or self._channel is None:
            raise IOError("RFCOMM channel is not open")

        # Convert to NSData for IOBluetooth
        ns_data = Foundation.NSData.alloc().initWithBytes_length_(data, len(data))

        # writeSync blocks until the data is sent
        result = self._channel.writeSync_length_(data, len(data))

        if result != 0:
            logger.error(f"RFCOMM write failed with error: {result}")
            # Try NSData approach as fallback
            try:
                result = self._channel.writeData_(ns_data)
                if result != 0:
                    raise IOError(f"RFCOMM write failed (error {result})")
            except Exception:
                raise IOError(f"RFCOMM write failed (error {result})")

        logger.debug(f"RFCOMM TX: {len(data)} bytes: {data.hex(' ')}")
        return len(data)

    def flush(self):
        """Flush write buffer. No-op for RFCOMM (writes are synchronous)."""
        pass

    def close(self):
        """Close the RFCOMM channel and clean up."""
        if self._channel is not None:
            try:
                self._channel.closeChannel()
            except Exception as e:
                logger.debug(f"Error closing channel: {e}")
            self._channel = None

        if self._device is not None and self._device.isConnected():
            try:
                self._device.closeConnection()
            except Exception as e:
                logger.debug(f"Error closing connection: {e}")
            self._device = None

        if self._runloop_thread is not None:
            self._runloop_thread.stop()
            self._runloop_thread.join(timeout=3.0)
            self._runloop_thread = None

        self._is_open = False
        self._delegate = None
        logger.info("RFCOMM bridge closed")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Scanning utility
# ---------------------------------------------------------------------------
def scan_for_bioradio(timeout: float = 10.0) -> Optional[str]:
    """
    Scan paired Bluetooth devices for one matching 'BioRadio'.
    Returns the Bluetooth address string, or None.
    """
    try:
        paired = IOBluetooth.IOBluetoothDevice.pairedDevices()
        if paired is None:
            logger.info("No paired Bluetooth devices found")
            return None

        for device in paired:
            name = device.name() or ""
            addr = device.addressString() or ""
            if "bioradio" in name.lower():
                logger.info(f"Found BioRadio: {name} ({addr})")
                return addr
            # Also check for device names like "AYA", "AVA"
            if any(tag in name.lower() for tag in ["aya", "ava", "biocapture"]):
                logger.info(f"Found BioRadio-like device: {name} ({addr})")
                return addr

        logger.info("No BioRadio found among paired devices")
        return None

    except Exception as e:
        logger.error(f"Error scanning paired devices: {e}")
        return None


# ---------------------------------------------------------------------------
# SDP-only diagnostic (no RFCOMM open attempt)
# ---------------------------------------------------------------------------
def sdp_discover(address: str):
    """
    Perform SDP discovery on a device and print all services.
    Useful for diagnosing what profiles/channels the device exposes.
    """
    print(f"\n  Performing SDP discovery on {address} ...")

    device = IOBluetooth.IOBluetoothDevice.withAddressString_(address)
    if device is None:
        print(f"  ERROR: Could not find device {address}")
        return

    print(f"  Device name: {device.name()}")
    print(f"  Connected: {device.isConnected()}")

    # Open connection if needed
    if not device.isConnected():
        print(f"  Opening baseband connection ...")
        result = device.openConnection()
        if result != 0:
            print(f"  ERROR: Failed to open connection (error {result})")
            return
        time.sleep(1.0)

    # Perform fresh SDP query
    print(f"  Running SDP query (this may take a few seconds) ...")
    result = device.performSDPQuery_(None)
    print(f"  SDP query result: {result} ({'OK' if result == 0 else 'ERROR'})")
    time.sleep(3.0)

    # Enumerate services
    services = device.services()
    if services is None or len(services) == 0:
        print(f"  No SDP services found on device.")
        print(f"\n  This means the BioRadio is not advertising ANY Bluetooth services.")
        print(f"  Possible causes:")
        print(f"    - BioRadio firmware doesn't support SDP (unlikely)")
        print(f"    - Bluetooth connection is at ACL level only")
        print(f"    - Device needs to be power-cycled")
        return

    print(f"\n  Found {len(services)} SDP service(s):\n")

    spp_uuid = IOBluetooth.IOBluetoothSDPUUID.uuid16_(0x1101)
    rfcomm_uuid = IOBluetooth.IOBluetoothSDPUUID.uuid16_(0x0003)

    for i, svc in enumerate(services):
        svc_name = "unnamed"
        try:
            name = svc.getServiceName()
            if name:
                svc_name = str(name)
        except Exception:
            pass

        print(f"  [{i+1}] Service: '{svc_name}'")

        # Check for RFCOMM channel
        has_rfcomm = False
        try:
            res, ch_id = svc.getRFCOMMChannelID_(None)
            if res == 0:
                print(f"      RFCOMM channel: {ch_id}")
                has_rfcomm = True
        except TypeError:
            try:
                res, ch_id = svc.getRFCOMMChannelID_()
                if res == 0:
                    print(f"      RFCOMM channel: {ch_id}")
                    has_rfcomm = True
            except Exception:
                pass
        except Exception:
            pass

        if not has_rfcomm:
            print(f"      RFCOMM channel: none")

        # Check for L2CAP PSM
        try:
            res, psm = svc.getL2CAPPSM_(None)
            if res == 0:
                print(f"      L2CAP PSM: {psm}")
        except Exception:
            pass

        # Check if it matches SPP UUID
        try:
            if svc.hasServiceFromArray_([spp_uuid]):
                print(f"      ** Matches SPP UUID (0x1101) **")
        except Exception:
            pass

        # Try to get service class UUIDs from the record
        try:
            # Attribute 0x0001 = ServiceClassIDList
            attr = svc.getAttributeDataElement_(0x0001)
            if attr:
                print(f"      ServiceClassIDList: {attr}")
        except Exception:
            pass

        print()

    # Close connection
    try:
        device.closeConnection()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="RFCOMM Bridge for BioRadio")
    parser.add_argument("address", nargs="?", default=None,
                        help="Bluetooth address (e.g., EC-FE-7E-12-BA-36)")
    parser.add_argument("--sdp-only", action="store_true",
                        help="Only run SDP discovery (don't try to open RFCOMM)")
    parser.add_argument("--channel", "-c", type=int, default=1,
                        help="RFCOMM channel ID to try (default: auto-discover)")
    args = parser.parse_args()

    print("=" * 60)
    print("  RFCOMM Bridge Test")
    print("=" * 60)

    # Step 1: Find the BioRadio
    address = args.address
    if not address:
        print("\n[1] Scanning for BioRadio among paired devices ...")
        address = scan_for_bioradio()

    if not address:
        print("  No BioRadio found. Make sure it's paired in Bluetooth settings.")
        print("  You can also specify an address directly:")
        print("    python rfcomm_bridge.py EC-FE-7E-12-BA-36")
        sys.exit(1)

    print(f"  Found BioRadio at: {address}")

    # SDP-only mode
    if args.sdp_only:
        sdp_discover(address)
        sys.exit(0)

    # Step 2: Open RFCOMM channel
    print(f"\n[2] Opening RFCOMM channel (with SDP discovery) ...")
    bridge = RFCOMMSerialBridge(address, channel_id=args.channel, timeout=3.0)

    try:
        bridge.open(connect_timeout=15.0)
        print(f"  Connected! RFCOMM channel {bridge._channel_id} is open.")
    except ConnectionError as e:
        print(f"\n  Connection failed: {e}")
        print(f"\n  Suggestion: Try SDP-only discovery to see what the device offers:")
        print(f"    python rfcomm_bridge.py --sdp-only {address}")
        sys.exit(1)

    # Step 3: Send firmware query
    CMD_GET_FIRMWARE = bytes([0xF0, 0xF1, 0x00])
    print(f"\n[3] Sending firmware query: {CMD_GET_FIRMWARE.hex(' ')}")

    bridge.write(CMD_GET_FIRMWARE)
    time.sleep(0.5)

    print(f"  Bytes waiting: {bridge.in_waiting}")
    response = bridge.read(64)

    if response:
        print(f"  GOT RESPONSE ({len(response)} bytes): {response.hex(' ')}")
    else:
        print(f"  No response received")

        # Try a few more times
        for i in range(3):
            print(f"\n  Retry {i+1}/3 ...")
            bridge.write(CMD_GET_FIRMWARE)
            time.sleep(1.0)
            response = bridge.read(64)
            if response:
                print(f"  GOT RESPONSE ({len(response)} bytes): {response.hex(' ')}")
                break
            else:
                print(f"  No response")

    # Step 4: Clean up
    print(f"\n[4] Closing connection ...")
    bridge.close()
    print("  Done.")
