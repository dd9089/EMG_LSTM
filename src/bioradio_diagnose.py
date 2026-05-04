#!/usr/bin/env python3
"""
BioRadio Port Diagnostic Tool
==============================

Systematically tests serial ports to find which one communicates with
the BioRadio device. Tries different stabilization delays, multiple
probe attempts, and reports raw hex responses.

Usage:
    python bioradio_diagnose.py                  # Auto-scan and diagnose
    python bioradio_diagnose.py --port /dev/cu.BioRadioAYA   # Test specific port
    python bioradio_diagnose.py --all            # Test ALL serial ports (not just candidates)

Cross-platform: Windows (COMx), macOS (/dev/cu.*, /dev/tty.*), Linux.
"""

import sys
import time
import argparse
from typing import Optional, List, Tuple, Dict

import serial
import serial.tools.list_ports

# ---------------------------------------------------------------------------
# Constants (must match bioradio.py)
# ---------------------------------------------------------------------------
BAUD_RATE = 460800
SYNC_BYTE = 0xF0

IS_MACOS = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"

# GetGlobal FirmwareVersion command: [sync=0xF0] [header=0xF1] [data=0x00]
# Header = 0xF0 (GetGlobal) | 0x01 (data length=1) = 0xF1
CMD_GET_FIRMWARE = bytes([SYNC_BYTE, 0xF1, 0x00])

# Stabilization delays to test (seconds)
STABILIZATION_DELAYS = [0.25, 0.50, 1.0, 2.0]

# Number of send attempts per stabilization delay
MAX_SEND_ATTEMPTS = 5

# Timeout per read attempt (seconds)
READ_TIMEOUT = 2.0

# DTR/RTS configurations to try
SIGNAL_CONFIGS = [
    {"dtr": False, "rts": False, "label": "DTR=off  RTS=off"},
    {"dtr": True,  "rts": False, "label": "DTR=on   RTS=off"},
    {"dtr": False, "rts": True,  "label": "DTR=off  RTS=on"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def list_all_ports() -> List[dict]:
    """List all serial ports with metadata."""
    ports = []
    for p in serial.tools.list_ports.comports():
        ports.append({
            "device": p.device,
            "name": p.name or "",
            "description": p.description or "",
            "hwid": p.hwid or "",
            "vid": p.vid,
            "pid": p.pid,
            "serial_number": p.serial_number or "",
            "manufacturer": p.manufacturer or "",
            "product": p.product or "",
        })
    return sorted(ports, key=lambda p: p["device"])


def is_bioradio_candidate(port: dict) -> bool:
    """Check if a port looks like it might be a BioRadio."""
    search_names = ["bioradio", "aya", "ava", "biocapture"]
    search_text = (port["device"] + " " + port["description"] + " " +
                   port["hwid"] + " " + port["manufacturer"] + " " +
                   port["product"]).lower()

    if any(name in search_text for name in search_names):
        return True

    # On Windows, any COM port could be BioRadio
    if IS_WINDOWS and port["device"].startswith("COM"):
        return True

    # On macOS, any non-builtin /dev/cu.* or /dev/tty.* could be BioRadio
    if IS_MACOS:
        dev = port["device"]
        skip = ["debug", "mals", "wlan", "usbmodem", "bluetooth"]
        if dev.startswith("/dev/cu.") or dev.startswith("/dev/tty."):
            if not any(s in dev.lower() for s in skip):
                return True

    return False


def try_open_port(port_name: str, dtr: bool = False, rts: bool = False,
                  timeout: float = READ_TIMEOUT) -> Optional[serial.Serial]:
    """Try to open a serial port. Returns Serial object or None."""
    try:
        ser = serial.Serial(
            port=port_name,
            baudrate=BAUD_RATE,
            timeout=timeout,
            write_timeout=timeout,
            rtscts=False,
            dsrdtr=False,
        )
        try:
            ser.dtr = dtr
            ser.rts = rts
        except Exception:
            pass  # Some platforms don't support these
        return ser
    except serial.SerialException as e:
        print(f"      [FAIL] Cannot open: {e}")
        return None
    except OSError as e:
        print(f"      [FAIL] OS error: {e}")
        return None


def drain_port(ser: serial.Serial) -> int:
    """Drain any stale data. Returns number of bytes drained."""
    drained = 0
    try:
        if ser.in_waiting:
            stale = ser.read(ser.in_waiting)
            drained = len(stale)
    except (OSError, serial.SerialException):
        pass
    return drained


def send_and_read(ser: serial.Serial, cmd: bytes,
                  timeout: float = READ_TIMEOUT) -> Tuple[Optional[bytes], float]:
    """
    Send a command and read whatever comes back.

    Returns:
        (response_bytes or None, response_time_seconds)
    """
    try:
        ser.write(cmd)
        ser.flush()
    except serial.SerialTimeoutException:
        return None, 0.0
    except (serial.SerialException, OSError) as e:
        print(f"        Write error: {e}")
        return None, 0.0

    t_start = time.monotonic()
    response = bytearray()
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        old_timeout = ser.timeout
        ser.timeout = min(0.5, remaining)
        try:
            byte = ser.read(1)
            if byte:
                response.extend(byte)
                # Tiny delay to let more bytes arrive
                time.sleep(0.01)
                try:
                    waiting = ser.in_waiting
                    if waiting > 0:
                        response.extend(ser.read(waiting))
                except OSError:
                    pass

                # If we have a sync byte and at least a header, check completeness
                if SYNC_BYTE in response and len(response) >= 3:
                    # Try to figure out expected length
                    sync_idx = response.index(SYNC_BYTE)
                    if sync_idx + 1 < len(response):
                        header = response[sync_idx + 1]
                        data_len = header & 0x0F
                        if data_len < 6:
                            expected = sync_idx + 2 + data_len  # sync + header + data
                        else:
                            # Extended length: need one more byte
                            if sync_idx + 2 < len(response):
                                expected = sync_idx + 3 + response[sync_idx + 2]
                            else:
                                continue  # Not enough data yet
                        if len(response) >= expected:
                            break  # Complete response!
        except (serial.SerialException, OSError):
            break
        finally:
            ser.timeout = old_timeout

    elapsed = time.monotonic() - t_start
    if response:
        return bytes(response), elapsed
    return None, elapsed


def parse_firmware_response(data: bytes) -> Optional[str]:
    """Try to extract firmware version from a response."""
    if not data or len(data) < 3:
        return None

    # Find the sync byte
    for i in range(len(data)):
        if data[i] == SYNC_BYTE and i + 1 < len(data):
            header = data[i + 1]
            cmd = header & 0xF0
            length = header & 0x0F

            # GetGlobal response should echo back as 0xF0 command
            if cmd == 0xF0 and length >= 1:
                # Check if this is a firmware version response
                # Data starts at i + 2, first byte is the param ID echo (0x00)
                data_start = i + 2
                if length >= 6 and data_start + 6 <= len(data):
                    # Expected: [param_id=0x00] [??] [fw_major] [fw_minor] [hw_major] [hw_minor]
                    if data[data_start] == 0x00:
                        fw = f"{data[data_start + 2]}.{data[data_start + 3]:02d}"
                        hw = f"{data[data_start + 4]}.{data[data_start + 5]:02d}"
                        return f"FW={fw} HW={hw}"
            elif length == 6:
                # Extended length
                if i + 2 < len(data):
                    ext_len = data[i + 2]
                    data_start = i + 3
                    if ext_len >= 6 and data_start + ext_len <= len(data):
                        if data[data_start] == 0x00:
                            fw = f"{data[data_start + 2]}.{data[data_start + 3]:02d}"
                            hw = f"{data[data_start + 4]}.{data[data_start + 5]:02d}"
                            return f"FW={fw} HW={hw}"

    return None


# ---------------------------------------------------------------------------
# Main Diagnostic
# ---------------------------------------------------------------------------
def diagnose_port(port_name: str, quick: bool = False) -> Dict:
    """
    Run full diagnostics on a single port.

    Returns a dict with results.
    """
    results = {
        "port": port_name,
        "open": False,
        "responded": False,
        "firmware": None,
        "best_delay": None,
        "best_attempt": None,
        "best_signals": None,
        "all_responses": [],
    }

    signal_configs = SIGNAL_CONFIGS if not quick else [SIGNAL_CONFIGS[0]]
    delays = STABILIZATION_DELAYS if not quick else [0.50, 1.0]
    max_attempts = MAX_SEND_ATTEMPTS if not quick else 3

    for sig_cfg in signal_configs:
        print(f"\n    --- Signals: {sig_cfg['label']} ---")

        ser = try_open_port(port_name, dtr=sig_cfg["dtr"], rts=sig_cfg["rts"])
        if ser is None:
            continue
        results["open"] = True

        for delay in delays:
            print(f"\n      Stabilization delay: {delay:.2f}s ...")
            time.sleep(delay)

            # Drain stale data
            drained = drain_port(ser)
            if drained > 0:
                print(f"      Drained {drained} stale bytes")

            for attempt in range(1, max_attempts + 1):
                print(f"      Attempt {attempt}/{max_attempts}: TX {CMD_GET_FIRMWARE.hex(' ')} ... ",
                      end="", flush=True)

                response, elapsed = send_and_read(ser, CMD_GET_FIRMWARE, timeout=READ_TIMEOUT)

                if response:
                    hex_str = response.hex(' ')
                    fw_info = parse_firmware_response(response)
                    status = f"[OK] ({len(response)}B in {elapsed:.3f}s)"
                    if fw_info:
                        status += f" -> {fw_info}"
                    print(status)
                    print(f"        RX: {hex_str}")

                    results["all_responses"].append({
                        "signals": sig_cfg["label"],
                        "delay": delay,
                        "attempt": attempt,
                        "response": hex_str,
                        "elapsed": elapsed,
                        "firmware": fw_info,
                    })

                    if not results["responded"]:
                        results["responded"] = True
                        results["firmware"] = fw_info
                        results["best_delay"] = delay
                        results["best_attempt"] = attempt
                        results["best_signals"] = sig_cfg["label"]

                    # Once we get a response, no need for more attempts at this delay
                    if quick:
                        break
                else:
                    print(f"[--] No response ({elapsed:.3f}s)")

                # Small pause between attempts
                time.sleep(0.05)

            # If we already got a response and running quick mode, skip other delays
            if results["responded"] and quick:
                break

        try:
            ser.close()
        except Exception:
            pass

        # If we got a response, skip other signal configs in quick mode
        if results["responded"] and quick:
            break

    return results


def main():
    parser = argparse.ArgumentParser(
        description="BioRadio Port Diagnostic Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bioradio_diagnose.py                           # Auto-scan and diagnose
  python bioradio_diagnose.py --port /dev/cu.BioRadioAYA  # Test specific port
  python bioradio_diagnose.py --port COM9               # Test specific Windows port
  python bioradio_diagnose.py --all                     # Test ALL serial ports
  python bioradio_diagnose.py --quick                   # Fast scan (fewer attempts)
        """
    )
    parser.add_argument("--port", "-p", default=None,
                        help="Test a specific port only")
    parser.add_argument("--all", action="store_true",
                        help="Test ALL serial ports, not just BioRadio candidates")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick mode: fewer delays and attempts")
    args = parser.parse_args()

    print("=" * 70)
    print("  BioRadio Port Diagnostic Tool")
    print(f"  Platform: {'macOS' if IS_MACOS else 'Windows' if IS_WINDOWS else 'Linux'}")
    print(f"  Baud Rate: {BAUD_RATE}")
    print(f"  Command: GetGlobal FirmwareVersion ({CMD_GET_FIRMWARE.hex(' ')})")
    print("=" * 70)

    # Step 1: List all serial ports
    print("\n[1] Enumerating serial ports...\n")
    all_ports = list_all_ports()

    if not all_ports:
        print("  No serial ports found!")
        print("  Make sure the BioRadio is powered on and paired via Bluetooth.")
        if IS_MACOS:
            print("  Try: ls /dev/cu.* /dev/tty.*")
        return

    for p in all_ports:
        candidate = is_bioradio_candidate(p)
        marker = " <-- candidate" if candidate else ""
        print(f"  {p['device']}{marker}")
        if p["description"] and p["description"] != "n/a":
            print(f"    Description: {p['description']}")
        if p["manufacturer"]:
            print(f"    Manufacturer: {p['manufacturer']}")
        if p["hwid"] and p["hwid"] != "n/a":
            print(f"    HWID: {p['hwid']}")

    # Step 2: Determine which ports to test
    if args.port:
        test_ports = [args.port]
        # On macOS, if user gave a tty.* port, also test the cu.* equivalent
        if IS_MACOS and "/dev/tty." in args.port:
            cu_equiv = args.port.replace("/dev/tty.", "/dev/cu.")
            test_ports.insert(0, cu_equiv)
            print(f"\n  Note: Also testing cu.* equivalent: {cu_equiv}")
        elif IS_MACOS and "/dev/cu." in args.port:
            tty_equiv = args.port.replace("/dev/cu.", "/dev/tty.")
            test_ports.append(tty_equiv)
            print(f"\n  Note: Also testing tty.* equivalent: {tty_equiv}")
    elif args.all:
        test_ports = [p["device"] for p in all_ports]
    else:
        # Auto-detect candidates
        candidates = [p["device"] for p in all_ports if is_bioradio_candidate(p)]
        if not candidates:
            print("\n  No BioRadio candidates found among serial ports.")
            print("  Use --all to test every port, or --port to specify one.")
            return

        # On macOS, prioritize cu.* over tty.*
        if IS_MACOS:
            candidates.sort(key=lambda p: (0 if "/dev/cu." in p else 1))

        test_ports = candidates

    # Step 3: Diagnose each port
    print(f"\n[2] Testing {len(test_ports)} port(s)...\n")
    all_results = []

    for port_name in test_ports:
        print(f"\n{'='*60}")
        print(f"  Testing: {port_name}")
        print(f"{'='*60}")

        result = diagnose_port(port_name, quick=args.quick)
        all_results.append(result)

    # Step 4: Summary
    print(f"\n\n{'='*70}")
    print("  DIAGNOSIS SUMMARY")
    print(f"{'='*70}\n")

    any_success = False
    best_port = None

    for r in all_results:
        status_icon = "[OK]" if r["responded"] else "[FAIL]" if r["open"] else "[SKIP]"
        print(f"  {status_icon} {r['port']}")

        if not r["open"]:
            print(f"        Could not open port")
        elif not r["responded"]:
            print(f"        Port opened but no BioRadio response")
        else:
            any_success = True
            print(f"        {r['firmware'] or 'Response received'}")
            print(f"        Best config: {r['best_signals']}, "
                  f"delay={r['best_delay']:.2f}s, "
                  f"attempt #{r['best_attempt']}")
            print(f"        Total successful responses: {len(r['all_responses'])}")

            if best_port is None:
                best_port = r

    if any_success and best_port:
        print(f"\n  RECOMMENDATION:")
        print(f"  ===============")
        print(f"  Use port: {best_port['port']}")
        if best_port["firmware"]:
            print(f"  Device:   {best_port['firmware']}")
        print(f"\n  Connect with:")
        print(f"    python bioradio.py --port {best_port['port']} --info")
        print(f"    python bioradio.py --port {best_port['port']} --lsl")

        if best_port["best_delay"] and best_port["best_delay"] > 0.5:
            print(f"\n  NOTE: Device needed {best_port['best_delay']:.2f}s stabilization delay.")
            print(f"  If connection is flaky, the BT link may need more time to settle.")
            print(f"  Attempt #{best_port['best_attempt']} succeeded (first {best_port['best_attempt']-1} failed).")
    elif not any_success:
        print(f"\n  No BioRadio responded on any tested port.")
        print(f"\n  Troubleshooting:")
        print(f"    1. Make sure the BioRadio is powered ON (LED blinking)")
        print(f"    2. Make sure it is paired via Bluetooth")
        if IS_MACOS:
            print(f"    3. Check System Settings > Bluetooth")
            print(f"    4. Try: ls /dev/cu.* /dev/tty.*")
            print(f"    5. Use /dev/cu.* ports (tty.* blocks on carrier detect)")
            print(f"    6. Try unpairing and re-pairing the device")
            print(f"    7. Try power cycling the BioRadio")
        elif IS_WINDOWS:
            print(f"    3. Check Device Manager > Ports (COM & LPT)")
            print(f"    4. Try the LOWER-numbered COM port")
            print(f"    5. Try removing and re-pairing in Bluetooth settings")

    print()


if __name__ == "__main__":
    main()
