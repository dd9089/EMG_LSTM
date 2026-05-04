#!/usr/bin/env python3
"""
BioRadio macOS Bluetooth Serial Diagnostic
============================================

Deep diagnostic for macOS Bluetooth serial issues where the port opens
but the BioRadio doesn't respond. Tests:

1. Basic port health (open/close, read/write capability)
2. Baud rate variations (macOS BT may ignore or require specific rates)
3. Flow control variations (xonxoff, rtscts, dsrdtr)
4. Port wake-up cycle (open-close-reopen trick)
5. tty.* vs cu.* comparison
6. Exclusive access / lock file detection
7. IOKit Bluetooth channel probing
8. Raw byte loopback detection
9. Alternative command bytes
10. Longer warm-up with keep-alive writes

Usage:
    python bioradio_diagnose_mac.py
    python bioradio_diagnose_mac.py --port /dev/cu.BioRadioAYA
"""

import sys
import os
import time
import struct
import argparse
import subprocess
from typing import Optional, List, Tuple

import serial
import serial.tools.list_ports

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYNC_BYTE = 0xF0

# GetGlobal FirmwareVersion: [sync=0xF0] [header=0xF1] [data=0x00]
CMD_GET_FIRMWARE = bytes([0xF0, 0xF1, 0x00])

# GetMode command: [sync=0xF0] [header=0x30] — no data
CMD_GET_MODE = bytes([0xF0, 0x30])

# SetMode Idle: [sync=0xF0] [header=0x21] [mode=0x00]
CMD_SET_IDLE = bytes([0xF0, 0x21, 0x00])

# Baud rates to test
BAUD_RATES = [460800, 230400, 115200, 57600, 38400, 19200, 9600]

# Flow control configs
FLOW_CONFIGS = [
    {"rtscts": False, "dsrdtr": False, "xonxoff": False, "label": "No flow control"},
    {"rtscts": True,  "dsrdtr": False, "xonxoff": False, "label": "RTS/CTS hardware"},
    {"rtscts": False, "dsrdtr": True,  "xonxoff": False, "label": "DSR/DTR hardware"},
    {"rtscts": False, "dsrdtr": False, "xonxoff": True,  "label": "XON/XOFF software"},
]


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def subsection(title: str):
    print(f"\n  --- {title} ---\n")


def ok(msg: str):
    print(f"    [OK] {msg}")


def fail(msg: str):
    print(f"    [FAIL] {msg}")


def info(msg: str):
    print(f"    [INFO] {msg}")


def warn(msg: str):
    print(f"    [WARN] {msg}")


# ---------------------------------------------------------------------------
# Test 1: Port Health
# ---------------------------------------------------------------------------
def test_port_health(port_name: str) -> bool:
    """Basic port open/close and property check."""
    section("TEST 1: Port Health Check")

    # Check if device node exists
    if os.path.exists(port_name):
        ok(f"Device node {port_name} exists")
    else:
        fail(f"Device node {port_name} does NOT exist")
        return False

    # Check permissions
    try:
        stat = os.stat(port_name)
        mode = oct(stat.st_mode)
        info(f"Permissions: {mode}")
        if os.access(port_name, os.R_OK | os.W_OK):
            ok("Read/Write access confirmed")
        else:
            fail("No read/write access — may need to run as root or adjust permissions")
            return False
    except OSError as e:
        fail(f"Cannot stat device: {e}")
        return False

    # Check for lock files
    lock_paths = [
        f"/var/lock/LCK..{os.path.basename(port_name)}",
        f"/tmp/LCK..{os.path.basename(port_name)}",
    ]
    for lp in lock_paths:
        if os.path.exists(lp):
            warn(f"Lock file found: {lp}")
            try:
                with open(lp, 'r') as f:
                    pid = f.read().strip()
                warn(f"  Locked by PID: {pid}")
                # Check if that PID is still running
                try:
                    os.kill(int(pid), 0)
                    fail(f"  PID {pid} is still running — another process has the port!")
                except (OSError, ValueError):
                    info(f"  PID {pid} is NOT running — stale lock file")
            except Exception:
                pass
        else:
            ok(f"No lock file at {lp}")

    # Try to open and close the port
    try:
        ser = serial.Serial(port_name, baudrate=460800, timeout=1.0)
        ok(f"Port opened successfully (fd={ser.fileno()})")
        info(f"  Baud: {ser.baudrate}")
        info(f"  Bytesize: {ser.bytesize}")
        info(f"  Parity: {ser.parity}")
        info(f"  Stopbits: {ser.stopbits}")
        try:
            info(f"  CTS: {ser.cts}")
            info(f"  DSR: {ser.dsr}")
            info(f"  RI: {ser.ri}")
            info(f"  CD: {ser.cd}")
        except Exception:
            info("  Modem status lines not available")
        ser.close()
        ok("Port closed cleanly")
    except serial.SerialException as e:
        fail(f"Cannot open port: {e}")
        return False

    return True


# ---------------------------------------------------------------------------
# Test 2: Check for processes using the port
# ---------------------------------------------------------------------------
def test_exclusive_access(port_name: str):
    """Check if other processes have the port open."""
    section("TEST 2: Exclusive Access Check")

    try:
        result = subprocess.run(
            ["lsof", port_name],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            warn(f"Processes using {port_name}:")
            for line in result.stdout.strip().split('\n'):
                print(f"      {line}")
        else:
            ok(f"No other processes are using {port_name}")
    except FileNotFoundError:
        info("lsof not available")
    except subprocess.TimeoutExpired:
        warn("lsof timed out")


# ---------------------------------------------------------------------------
# Test 3: Bluetooth connection status
# ---------------------------------------------------------------------------
def test_bluetooth_status():
    """Check macOS Bluetooth status using system_profiler."""
    section("TEST 3: Bluetooth Status")

    try:
        result = subprocess.run(
            ["system_profiler", "SPBluetoothDataType"],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout

        # Look for BioRadio in the output
        lines = output.split('\n')
        in_bioradio_section = False
        bioradio_info = []
        indent_level = 0

        for line in lines:
            lower_line = line.lower()
            if 'bioradio' in lower_line or 'aya' in lower_line:
                in_bioradio_section = True
                indent_level = len(line) - len(line.lstrip())
                bioradio_info.append(line)
            elif in_bioradio_section:
                current_indent = len(line) - len(line.lstrip())
                if line.strip() and current_indent > indent_level:
                    bioradio_info.append(line)
                elif line.strip() and current_indent <= indent_level:
                    in_bioradio_section = False

        if bioradio_info:
            ok("BioRadio found in Bluetooth profile:")
            for bl in bioradio_info:
                print(f"      {bl.strip()}")
        else:
            warn("BioRadio NOT found in system_profiler Bluetooth output")
            info("This might just mean the name doesn't contain 'BioRadio'")

        # Check general BT status
        if "State: On" in output or "state: attrib_on" in output.lower():
            ok("Bluetooth is ON")
        if "Connected: Yes" in output:
            ok("Device shows 'Connected: Yes'")
        elif "Paired: Yes" in output:
            info("Device shows 'Paired: Yes' but may not be 'Connected'")

    except FileNotFoundError:
        info("system_profiler not available")
    except subprocess.TimeoutExpired:
        warn("system_profiler timed out")


# ---------------------------------------------------------------------------
# Test 4: Baud Rate Sweep
# ---------------------------------------------------------------------------
def test_baud_rates(port_name: str) -> Optional[int]:
    """Try different baud rates — some BT adapters ignore baud setting."""
    section("TEST 4: Baud Rate Sweep")

    working_baud = None

    for baud in BAUD_RATES:
        print(f"    Testing baud={baud} ... ", end="", flush=True)
        try:
            ser = serial.Serial(
                port=port_name,
                baudrate=baud,
                timeout=2.0,
                write_timeout=2.0,
                rtscts=False,
                dsrdtr=False,
            )
            try:
                ser.dtr = False
                ser.rts = False
            except Exception:
                pass

            time.sleep(0.5)

            # Drain
            try:
                if ser.in_waiting:
                    ser.read(ser.in_waiting)
            except Exception:
                pass

            # Send firmware query
            ser.write(CMD_GET_FIRMWARE)
            ser.flush()

            # Wait for response
            time.sleep(0.1)
            response = bytearray()
            deadline = time.monotonic() + 2.0

            while time.monotonic() < deadline:
                try:
                    if ser.in_waiting > 0:
                        response.extend(ser.read(ser.in_waiting))
                        if len(response) >= 3 and SYNC_BYTE in response:
                            break
                except Exception:
                    break
                time.sleep(0.05)

            ser.close()

            if response:
                ok(f"baud={baud} — GOT RESPONSE: {response.hex(' ')}")
                if working_baud is None:
                    working_baud = baud
            else:
                print(f"no response")

        except serial.SerialException as e:
            print(f"error: {e}")

    if working_baud:
        ok(f"Working baud rate found: {working_baud}")
    else:
        fail("No response at any baud rate")

    return working_baud


# ---------------------------------------------------------------------------
# Test 5: Flow Control Variations
# ---------------------------------------------------------------------------
def test_flow_control(port_name: str):
    """Try different flow control settings."""
    section("TEST 5: Flow Control Variations")

    for fc in FLOW_CONFIGS:
        print(f"    Testing {fc['label']} ... ", end="", flush=True)
        try:
            ser = serial.Serial(
                port=port_name,
                baudrate=460800,
                timeout=2.0,
                write_timeout=2.0,
                rtscts=fc["rtscts"],
                dsrdtr=fc["dsrdtr"],
                xonxoff=fc["xonxoff"],
            )
            time.sleep(0.5)

            try:
                if ser.in_waiting:
                    ser.read(ser.in_waiting)
            except Exception:
                pass

            ser.write(CMD_GET_FIRMWARE)
            ser.flush()

            response = bytearray()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                try:
                    if ser.in_waiting > 0:
                        response.extend(ser.read(ser.in_waiting))
                        if len(response) >= 3:
                            break
                except Exception:
                    break
                time.sleep(0.05)

            ser.close()

            if response:
                ok(f"{fc['label']} — GOT RESPONSE: {response.hex(' ')}")
            else:
                print(f"no response")

        except serial.SerialException as e:
            print(f"error: {e}")


# ---------------------------------------------------------------------------
# Test 6: Port Wake-Up Cycle
# ---------------------------------------------------------------------------
def test_wake_up_cycle(port_name: str):
    """
    Some macOS BT serial ports need an initial open-close-reopen cycle
    to properly initialize the RFCOMM channel.
    """
    section("TEST 6: Port Wake-Up Cycle (open-close-reopen)")

    for attempt in range(1, 4):
        print(f"\n    Wake-up cycle #{attempt}:")

        # Step 1: Open and immediately close
        try:
            print(f"      Step 1: Opening port briefly ...", end="", flush=True)
            ser = serial.Serial(port_name, baudrate=460800, timeout=0.5)
            time.sleep(0.25)
            ser.close()
            print(f" closed")
        except Exception as e:
            print(f" error: {e}")
            continue

        # Step 2: Wait
        wait = 0.5 * attempt
        print(f"      Step 2: Waiting {wait:.1f}s ...")
        time.sleep(wait)

        # Step 3: Reopen and send command
        try:
            print(f"      Step 3: Reopening and sending command ... ", end="", flush=True)
            ser = serial.Serial(
                port=port_name,
                baudrate=460800,
                timeout=2.0,
                write_timeout=2.0,
                rtscts=False,
                dsrdtr=False,
            )
            try:
                ser.dtr = False
                ser.rts = False
            except Exception:
                pass

            time.sleep(0.5)

            # Drain
            try:
                if ser.in_waiting:
                    ser.read(ser.in_waiting)
            except Exception:
                pass

            ser.write(CMD_GET_FIRMWARE)
            ser.flush()

            response = bytearray()
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                try:
                    if ser.in_waiting > 0:
                        response.extend(ser.read(ser.in_waiting))
                        if len(response) >= 3:
                            break
                except Exception:
                    break
                time.sleep(0.05)

            ser.close()

            if response:
                ok(f"Cycle #{attempt} — GOT RESPONSE: {response.hex(' ')}")
                return True
            else:
                print(f"no response")

        except serial.SerialException as e:
            print(f"error: {e}")

    fail("Wake-up cycle did not produce a response")
    return False


# ---------------------------------------------------------------------------
# Test 7: tty.* vs cu.* comparison
# ---------------------------------------------------------------------------
def test_tty_vs_cu(port_name: str):
    """Compare behavior of tty.* and cu.* variants."""
    section("TEST 7: tty.* vs cu.* Comparison")

    if "/dev/cu." in port_name:
        cu_port = port_name
        tty_port = port_name.replace("/dev/cu.", "/dev/tty.")
    elif "/dev/tty." in port_name:
        tty_port = port_name
        cu_port = port_name.replace("/dev/tty.", "/dev/cu.")
    else:
        info("Not a /dev/cu.* or /dev/tty.* port — skipping")
        return

    for pname, ptype in [(cu_port, "cu.*"), (tty_port, "tty.*")]:
        print(f"\n    Testing {ptype} ({pname}):")

        if not os.path.exists(pname):
            info(f"      {pname} does not exist — skipping")
            continue

        try:
            # Use a short timeout for tty.* since it blocks on carrier detect
            timeout = 2.0 if ptype == "cu.*" else 3.0
            print(f"      Opening with timeout={timeout}s ... ", end="", flush=True)

            ser = serial.Serial(
                port=pname,
                baudrate=460800,
                timeout=timeout,
                write_timeout=timeout,
                rtscts=False,
                dsrdtr=False,
            )
            print(f"opened")

            try:
                ser.dtr = False
                ser.rts = False
            except Exception:
                pass

            time.sleep(0.5)

            try:
                if ser.in_waiting:
                    stale = ser.read(ser.in_waiting)
                    info(f"      Drained {len(stale)} stale bytes")
            except Exception:
                pass

            print(f"      Sending firmware query ... ", end="", flush=True)
            ser.write(CMD_GET_FIRMWARE)
            ser.flush()

            response = bytearray()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    if ser.in_waiting > 0:
                        response.extend(ser.read(ser.in_waiting))
                        if len(response) >= 3:
                            break
                except Exception:
                    break
                time.sleep(0.05)

            ser.close()

            if response:
                ok(f"{ptype} — GOT RESPONSE: {response.hex(' ')}")
            else:
                print(f"no response")

        except serial.SerialException as e:
            fail(f"{ptype} — {e}")


# ---------------------------------------------------------------------------
# Test 8: Alternative Commands
# ---------------------------------------------------------------------------
def test_alternative_commands(port_name: str):
    """Try different BioRadio commands in case firmware version isn't supported."""
    section("TEST 8: Alternative Commands")

    commands = [
        ("GetGlobal FirmwareVersion", CMD_GET_FIRMWARE),
        ("GetMode", CMD_GET_MODE),
        ("SetMode Idle", CMD_SET_IDLE),
        ("Raw sync byte only", bytes([0xF0])),
        ("Double sync", bytes([0xF0, 0xF0])),
        ("Null bytes", bytes([0x00, 0x00, 0x00])),
    ]

    try:
        ser = serial.Serial(
            port=port_name,
            baudrate=460800,
            timeout=2.0,
            write_timeout=2.0,
            rtscts=False,
            dsrdtr=False,
        )
        try:
            ser.dtr = False
            ser.rts = False
        except Exception:
            pass

        time.sleep(1.0)

        for label, cmd in commands:
            # Drain
            try:
                if ser.in_waiting:
                    ser.read(ser.in_waiting)
            except Exception:
                pass

            print(f"    TX [{label}]: {cmd.hex(' ')} ... ", end="", flush=True)
            ser.write(cmd)
            ser.flush()

            response = bytearray()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                try:
                    if ser.in_waiting > 0:
                        response.extend(ser.read(ser.in_waiting))
                        if len(response) >= 1:
                            # Give a bit more time for rest of packet
                            time.sleep(0.1)
                            if ser.in_waiting > 0:
                                response.extend(ser.read(ser.in_waiting))
                            break
                except Exception:
                    break
                time.sleep(0.05)

            if response:
                ok(f"GOT RESPONSE ({len(response)}B): {response.hex(' ')}")
            else:
                print(f"no response")

            time.sleep(0.2)

        ser.close()

    except serial.SerialException as e:
        fail(f"Could not open port: {e}")


# ---------------------------------------------------------------------------
# Test 9: Long Warm-Up with Keep-Alive
# ---------------------------------------------------------------------------
def test_long_warmup(port_name: str):
    """
    Some BT serial adapters need multiple writes before the RFCOMM channel
    is truly bidirectional. Send periodic pings over a longer period.
    """
    section("TEST 9: Long Warm-Up (30s with periodic pings)")

    try:
        ser = serial.Serial(
            port=port_name,
            baudrate=460800,
            timeout=1.0,
            write_timeout=2.0,
            rtscts=False,
            dsrdtr=False,
        )
        try:
            ser.dtr = False
            ser.rts = False
        except Exception:
            pass

        info("Sending firmware query every 2s for 30s ...")
        info("(Watch the BioRadio LED for any change in blink pattern)")
        print()

        start = time.monotonic()
        ping_count = 0

        while time.monotonic() - start < 30.0:
            elapsed = time.monotonic() - start
            ping_count += 1

            # Drain
            try:
                if ser.in_waiting:
                    stale = ser.read(ser.in_waiting)
                    if stale:
                        ok(f"  t={elapsed:.1f}s ping #{ping_count}: RECEIVED {len(stale)}B: {stale.hex(' ')}")
                        # Keep reading
                        time.sleep(0.1)
                        while ser.in_waiting > 0:
                            more = ser.read(ser.in_waiting)
                            if more:
                                ok(f"    + {len(more)}B more: {more.hex(' ')}")
                            time.sleep(0.05)
                        ser.close()
                        return True
            except Exception:
                pass

            print(f"    t={elapsed:5.1f}s  ping #{ping_count:2d}: TX {CMD_GET_FIRMWARE.hex(' ')} ... ",
                  end="", flush=True)

            try:
                ser.write(CMD_GET_FIRMWARE)
                ser.flush()
            except Exception as e:
                print(f"write error: {e}")
                break

            # Wait for response
            response = bytearray()
            ping_deadline = time.monotonic() + 1.5
            while time.monotonic() < ping_deadline:
                try:
                    if ser.in_waiting > 0:
                        response.extend(ser.read(ser.in_waiting))
                        if len(response) >= 3:
                            break
                except Exception:
                    break
                time.sleep(0.05)

            if response:
                ok(f"RESPONSE ({len(response)}B): {response.hex(' ')}")
                ser.close()
                return True
            else:
                print("no response")

            # Wait before next ping
            time.sleep(0.5)

        ser.close()
        fail("No response after 30s of periodic pings")
        return False

    except serial.SerialException as e:
        fail(f"Could not open port: {e}")
        return False


# ---------------------------------------------------------------------------
# Test 10: macOS IOKit / Bluetooth RFCOMM Info
# ---------------------------------------------------------------------------
def test_iokit_info():
    """Gather IOKit information about Bluetooth serial services."""
    section("TEST 10: IOKit Bluetooth Serial Info")

    # Check Bluetooth RFCOMM channels
    try:
        result = subprocess.run(
            ["ioreg", "-l", "-w", "0"],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout.lower()

        # Count RFCOMM references
        rfcomm_count = output.count("rfcomm")
        spp_count = output.count("serial port")
        info(f"IORegistry RFCOMM references: {rfcomm_count}")
        info(f"IORegistry 'serial port' references: {spp_count}")

        # Look for BioRadio-specific entries
        lines = result.stdout.split('\n')
        for i, line in enumerate(lines):
            if 'bioradio' in line.lower() or 'aya' in line.lower():
                info(f"BioRadio-related IOKit entry:")
                # Print surrounding context
                start = max(0, i - 2)
                end = min(len(lines), i + 5)
                for j in range(start, end):
                    marker = ">>>" if j == i else "   "
                    print(f"      {marker} {lines[j].strip()}")

    except FileNotFoundError:
        info("ioreg not available")
    except subprocess.TimeoutExpired:
        warn("ioreg timed out")

    # Check if Bluetooth serial service is loaded
    try:
        result = subprocess.run(
            ["kextstat"],
            capture_output=True, text=True, timeout=10
        )
        bt_kexts = [line for line in result.stdout.split('\n')
                    if 'bluetooth' in line.lower() or 'serial' in line.lower()]
        if bt_kexts:
            info("Bluetooth/Serial kernel extensions loaded:")
            for k in bt_kexts:
                parts = k.split()
                if len(parts) >= 6:
                    print(f"      {parts[-1]}")
        else:
            warn("No Bluetooth kernel extensions found in kextstat")
    except FileNotFoundError:
        info("kextstat not available (may need to use kmutil on newer macOS)")
        # Try kmutil on newer macOS
        try:
            result = subprocess.run(
                ["kmutil", "showloaded", "--list-only"],
                capture_output=True, text=True, timeout=10
            )
            bt_entries = [line for line in result.stdout.split('\n')
                          if 'bluetooth' in line.lower() or 'serial' in line.lower()]
            if bt_entries:
                info("Bluetooth/Serial kexts (via kmutil):")
                for k in bt_entries:
                    print(f"      {k.strip()}")
        except Exception:
            info("kmutil also not available")
    except subprocess.TimeoutExpired:
        warn("kextstat timed out")


# ---------------------------------------------------------------------------
# Test 11: Screen / cu tool test
# ---------------------------------------------------------------------------
def test_screen_hint(port_name: str):
    """Suggest using screen or cu as an alternative verification."""
    section("TEST 11: Manual Verification Hints")

    info("If all automated tests fail, try manual serial tools:")
    print()
    print(f"    Option A — screen (Ctrl-A then K to quit):")
    print(f"      screen {port_name} 460800")
    print()
    print(f"    Option B — cu:")
    print(f"      cu -l {port_name} -s 460800")
    print()
    print(f"    Option C — Python REPL:")
    print(f"      python3 -c \"")
    print(f"        import serial, time")
    print(f"        s = serial.Serial('{port_name}', 460800, timeout=5)")
    print(f"        time.sleep(1)")
    print(f"        s.write(bytes([0xF0, 0xF1, 0x00]))")
    print(f"        s.flush()")
    print(f"        time.sleep(2)")
    print(f"        print('waiting:', s.in_waiting)")
    print(f"        print('data:', s.read(100).hex(' ') if s.in_waiting else 'nothing')\"")
    print()
    info("If 'screen' shows garbled text when you turn the BioRadio on,")
    info("that confirms the BT serial link IS passing data.")
    info("If screen shows nothing, the issue is at the macOS BT/RFCOMM level.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if sys.platform != "darwin":
        print("WARNING: This diagnostic is designed for macOS.")
        print("Some tests may not work on other platforms.")
        print()

    parser = argparse.ArgumentParser(description="BioRadio macOS BT Serial Diagnostic")
    parser.add_argument("--port", "-p", default=None,
                        help="Port to test (default: auto-detect /dev/cu.BioRadio*)")
    parser.add_argument("--skip-long", action="store_true",
                        help="Skip the 30-second long warm-up test")
    args = parser.parse_args()

    print("=" * 70)
    print("  BioRadio macOS Bluetooth Serial Deep Diagnostic")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  pyserial: {serial.__version__}")
    print(f"  Platform: {sys.platform}")
    print("=" * 70)

    # Find port
    port_name = args.port
    if not port_name:
        # Auto-detect
        for p in serial.tools.list_ports.comports():
            if "bioradio" in p.device.lower() and "/dev/cu." in p.device:
                port_name = p.device
                break

    if not port_name:
        # Try listing /dev/cu.BioRadio* directly
        import glob
        matches = sorted(glob.glob("/dev/cu.BioRadio*"))
        if matches:
            port_name = matches[0]

    if not port_name:
        print("\nERROR: No BioRadio port found.")
        print("Use --port /dev/cu.YourPort to specify manually.")
        print("\nAvailable ports:")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}  ({p.description})")
        return

    print(f"\n  Target port: {port_name}\n")

    # Run tests
    if not test_port_health(port_name):
        print("\n  Port health check failed — cannot continue.")
        return

    test_exclusive_access(port_name)
    test_bluetooth_status()
    test_baud_rates(port_name)
    test_flow_control(port_name)
    test_wake_up_cycle(port_name)
    test_tty_vs_cu(port_name)
    test_alternative_commands(port_name)

    if not args.skip_long:
        test_long_warmup(port_name)
    else:
        info("Skipping long warm-up test (--skip-long)")

    test_iokit_info()
    test_screen_hint(port_name)

    # Summary
    section("DIAGNOSTIC COMPLETE")
    info("Review the results above to identify the issue.")
    print()
    info("Most common macOS BT serial issues:")
    print("    1. RFCOMM channel not actually established (BT shows 'Connected'")
    print("       but SPP/RFCOMM serial service isn't active)")
    print("    2. Need to pair with specific BT serial profile, not just generic")
    print("    3. macOS Bluetooth stack caches stale connections — try:")
    print("       a. Turn BioRadio OFF")
    print("       b. Remove (Forget) it from System Settings > Bluetooth")
    print("       c. Restart Bluetooth: sudo pkill bluetoothd")
    print("       d. Turn BioRadio ON")
    print("       e. Re-pair from System Settings > Bluetooth")
    print("       f. Run this diagnostic again")
    print()
    info("If NONE of the tests get a response, the issue is likely that")
    info("macOS is pairing at the ACL (low-level) layer but NOT establishing")
    info("an RFCOMM serial channel. The BioRadio uses Bluetooth SPP (Serial")
    info("Port Profile) which requires RFCOMM. Some macOS versions have")
    info("trouble auto-negotiating SPP — the device pairs but serial data")
    info("never flows.")


if __name__ == "__main__":
    main()
