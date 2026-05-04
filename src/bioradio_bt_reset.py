#!/usr/bin/env python3
"""
BioRadio Bluetooth Nuclear Reset & Re-pair Diagnostic
=======================================================

After extensive testing, we found that macOS Sonoma creates a phantom serial
port (/dev/cu.BioRadioAYA) from PersistentPorts config, but no actual RFCOMM
data channel exists behind it. Evidence:
  - MaxACLPacketSize = 0 (ACL data path never negotiated)
  - Link Level Encryption = 0 (BioRadio likely requires encryption)
  - Modem signals (DSR/CTS/CD) all True but hardcoded by pseudo-serial driver
  - Zero data in either direction at any baud rate / flow control

This script performs a complete Bluetooth reset, guides through re-pairing,
then immediately verifies the connection state at every layer.

Usage:
    python bioradio_bt_reset.py              # Full reset + diagnose
    python bioradio_bt_reset.py --check-only # Just check current state
    python bioradio_bt_reset.py --hci        # Attempt HCI-level encryption fix

Requires: pyserial, pyobjc-framework-IOBluetooth (optional for HCI fix)
"""

import sys
import os
import time
import subprocess
import argparse
import platform
import glob as glob_module

if sys.platform != "darwin":
    print("ERROR: This script is macOS-only")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BIORADIO_ADDR = "EC:FE:7E:12:BA:36"
BIORADIO_NAME_HINTS = ["bioradio", "aya", "ava", "biocapture"]

CMD_GET_FIRMWARE = bytes([0xF0, 0xF1, 0x00])


def run(cmd, timeout=10, check=False):
    """Run a shell command and return (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if check and r.returncode != 0:
            print(f"  [WARN] Command failed: {' '.join(cmd)}")
            print(f"         stderr: {r.stderr.strip()}")
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def ok(msg):
    print(f"  [OK]   {msg}")

def fail(msg):
    print(f"  [FAIL] {msg}")

def info(msg):
    print(f"  [INFO] {msg}")

def warn(msg):
    print(f"  [WARN] {msg}")


# ---------------------------------------------------------------------------
# Phase 1: Check current BT state
# ---------------------------------------------------------------------------
def check_current_state():
    """Check every layer of the Bluetooth connection."""
    section("PHASE 1: Current Bluetooth State")

    # 1a. Is Bluetooth on?
    rc, out, _ = run(["defaults", "read", "/Library/Preferences/com.apple.Bluetooth",
                       "ControllerPowerState"])
    bt_on = out.strip() == "1"
    if bt_on:
        ok("Bluetooth is ON")
    else:
        fail("Bluetooth is OFF — turn it on in System Settings")
        return False

    # 1b. Is BioRadio paired?
    rc, out, _ = run(["system_profiler", "SPBluetoothDataType"], timeout=15)
    bioradio_found = False
    connected = False
    encryption_line = ""
    services_line = ""

    lines = out.split('\n')
    in_bioradio = False
    indent = 0

    for i, line in enumerate(lines):
        lower = line.lower()
        if any(h in lower for h in BIORADIO_NAME_HINTS):
            in_bioradio = True
            indent = len(line) - len(line.lstrip())
            bioradio_found = True
            info(f"Found BioRadio entry: {line.strip()}")
        elif in_bioradio:
            cur_indent = len(line) - len(line.lstrip())
            if line.strip() and cur_indent <= indent and not line.strip().startswith(("Address", "Major", "Minor",
                                                                                       "Paired", "Connected", "Services",
                                                                                       "Firmware", "Vendor", "Link",
                                                                                       "RSSI")):
                in_bioradio = False
            elif "connected: yes" in lower:
                connected = True
            elif "link level encryption" in lower:
                encryption_line = line.strip()
            elif "services:" in lower:
                services_line = line.strip()

    if bioradio_found:
        ok("BioRadio is PAIRED")
        if connected:
            ok("BioRadio shows CONNECTED")
        else:
            warn("BioRadio shows NOT connected (may need to open serial port to trigger connection)")
        if encryption_line:
            info(f"  {encryption_line}")
        if services_line:
            info(f"  {services_line}")
    else:
        fail("BioRadio NOT found in paired devices")
        return False

    # 1c. Check serial port existence
    ports = sorted(glob_module.glob("/dev/cu.BioRadio*") + glob_module.glob("/dev/tty.BioRadio*"))
    if ports:
        ok(f"Serial port(s) exist: {', '.join(ports)}")
    else:
        fail("No /dev/cu.BioRadio* or /dev/tty.BioRadio* found")

    # 1d. Check PersistentPorts in BT plist
    rc, out, _ = run(["sudo", "/usr/libexec/PlistBuddy", "-c",
                       "Print :PersistentPorts", "/Library/Preferences/com.apple.Bluetooth.plist"],
                      timeout=5)
    if "BioRadio" in out or BIORADIO_ADDR.replace(":", "-").lower() in out.lower() or \
       BIORADIO_ADDR.lower() in out.lower():
        ok("PersistentPorts entry exists for BioRadio")
        for line in out.strip().split('\n'):
            if line.strip():
                info(f"  {line.strip()}")
    else:
        warn("No PersistentPorts entry for BioRadio")

    # 1e. Check IOKit driver state
    rc, out, _ = run(["ioreg", "-l", "-w", "0"], timeout=15)
    if "IOUserBluetoothSerialDriver" in out:
        ok("IOUserBluetoothSerialDriver is loaded")
    else:
        warn("IOUserBluetoothSerialDriver NOT found in IORegistry")

    # Check MaxACLPacketSize for BioRadio device
    bioradio_addr_nocolon = BIORADIO_ADDR.replace(":", "").lower()
    lines = out.split('\n')
    for i, line in enumerate(lines):
        if bioradio_addr_nocolon in line.lower():
            # Look at surrounding context for MaxACLPacketSize
            start = max(0, i - 5)
            end = min(len(lines), i + 20)
            for j in range(start, end):
                if "MaxACLPacketSize" in lines[j]:
                    val = lines[j].split("=")[-1].strip()
                    if val == "0":
                        fail(f"MaxACLPacketSize = 0 — ACL data path NOT established")
                    else:
                        ok(f"MaxACLPacketSize = {val} — ACL data path looks active")
                    break

    # 1f. Quick serial test
    try:
        import serial
        cu_port = None
        for p in sorted(glob_module.glob("/dev/cu.BioRadio*")):
            cu_port = p
            break

        if cu_port:
            info(f"Quick serial test on {cu_port} ...")
            try:
                ser = serial.Serial(cu_port, baudrate=460800, timeout=2.0,
                                    write_timeout=2.0, rtscts=False, dsrdtr=False)
                ser.dtr = False
                ser.rts = False
                time.sleep(0.5)

                # Drain
                if ser.in_waiting:
                    ser.read(ser.in_waiting)

                ser.write(CMD_GET_FIRMWARE)
                ser.flush()

                response = bytearray()
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    if ser.in_waiting > 0:
                        response.extend(ser.read(ser.in_waiting))
                        if len(response) >= 3:
                            break
                    time.sleep(0.05)

                ser.close()

                if response:
                    ok(f"*** SERIAL RESPONSE RECEIVED: {response.hex(' ')} ***")
                    ok("The BioRadio IS responding! The connection works!")
                    return True
                else:
                    fail("No serial response — RFCOMM data path is broken")
            except serial.SerialException as e:
                fail(f"Serial port error: {e}")
        else:
            info("No serial port to test")
    except ImportError:
        warn("pyserial not installed — skipping serial test")

    return bioradio_found


# ---------------------------------------------------------------------------
# Phase 2: Nuclear reset
# ---------------------------------------------------------------------------
def nuclear_reset():
    """Complete Bluetooth reset: forget device, clear plist, restart daemon."""
    section("PHASE 2: Nuclear Bluetooth Reset")

    print("  This will:")
    print("    1. Remove BioRadio from paired devices")
    print("    2. Clear the PersistentPorts entry")
    print("    3. Restart the Bluetooth daemon")
    print("    4. Guide you through re-pairing")
    print()

    # Confirm
    resp = input("  Proceed with nuclear reset? [y/N] ").strip().lower()
    if resp != 'y':
        info("Aborted")
        return False

    # Step 1: Try to remove via blueutil (if installed)
    rc, _, _ = run(["which", "blueutil"])
    has_blueutil = rc == 0

    if has_blueutil:
        info("Using blueutil to unpair...")
        addr_dash = BIORADIO_ADDR.replace(":", "-").lower()
        run(["blueutil", "--unpair", addr_dash])
        run(["blueutil", "--unpair", BIORADIO_ADDR.lower()])
        time.sleep(1)
    else:
        info("blueutil not installed (brew install blueutil) — using manual approach")

    # Step 2: Remove PersistentPorts entry
    info("Removing PersistentPorts entry for BioRadio...")
    # Try both address formats
    for addr_key in [BIORADIO_ADDR, BIORADIO_ADDR.replace(":", "-")]:
        run(["sudo", "/usr/libexec/PlistBuddy", "-c",
             f"Delete :PersistentPorts:{addr_key}",
             "/Library/Preferences/com.apple.Bluetooth.plist"])

    # Step 3: Remove from System Keychain (where Sonoma stores pairing keys)
    info("Removing BioRadio pairing from System Keychain...")
    # This requires sudo
    for name_hint in BIORADIO_NAME_HINTS:
        run(["sudo", "security", "delete-generic-password", "-l", name_hint,
             "/Library/Keychains/System.keychain"], timeout=5)

    # Also try by address
    run(["sudo", "security", "delete-generic-password", "-l", BIORADIO_ADDR,
         "/Library/Keychains/System.keychain"], timeout=5)

    # Step 4: Restart bluetoothd
    info("Restarting Bluetooth daemon...")
    run(["sudo", "pkill", "-HUP", "bluetoothd"])
    time.sleep(2)

    # If that didn't restart it, kill it hard (launchd will restart it)
    rc, out, _ = run(["pgrep", "bluetoothd"])
    if rc != 0:
        info("bluetoothd was killed, waiting for launchd to restart it...")
    else:
        info(f"bluetoothd still running (PID {out.strip()})")

    # Toggle BT off/on
    if has_blueutil:
        info("Toggling Bluetooth off/on with blueutil...")
        run(["blueutil", "--power", "0"])
        time.sleep(2)
        run(["blueutil", "--power", "1"])
        time.sleep(3)
    else:
        info("Toggling Bluetooth via defaults...")
        run(["sudo", "defaults", "write", "/Library/Preferences/com.apple.Bluetooth",
             "ControllerPowerState", "-int", "0"])
        run(["sudo", "pkill", "bluetoothd"])
        time.sleep(3)
        run(["sudo", "defaults", "write", "/Library/Preferences/com.apple.Bluetooth",
             "ControllerPowerState", "-int", "1"])
        run(["sudo", "pkill", "bluetoothd"])
        time.sleep(3)

    ok("Bluetooth reset complete")
    print()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  NOW: Re-pair the BioRadio                             │")
    print("  │                                                         │")
    print("  │  1. Make sure the BioRadio is powered ON                │")
    print("  │  2. Open System Settings → Bluetooth                    │")
    print("  │  3. Wait for 'BioRadioAYA' to appear under Nearby      │")
    print("  │  4. Click 'Connect' next to it                          │")
    print("  │  5. Wait for it to show 'Connected'                     │")
    print("  │  6. Press Enter here when done                          │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()
    input("  Press Enter after re-pairing the BioRadio...")

    return True


# ---------------------------------------------------------------------------
# Phase 3: Post-pair verification
# ---------------------------------------------------------------------------
def post_pair_verify():
    """Immediately check everything after a fresh pair."""
    section("PHASE 3: Post-Pair Verification")

    # Wait a moment for drivers to load
    info("Waiting 3 seconds for drivers to initialize...")
    time.sleep(3)

    # Check if serial port appeared
    ports = sorted(glob_module.glob("/dev/cu.BioRadio*"))
    attempts = 0
    while not ports and attempts < 5:
        attempts += 1
        info(f"  No serial port yet, waiting... (attempt {attempts}/5)")
        time.sleep(2)
        ports = sorted(glob_module.glob("/dev/cu.BioRadio*"))

    if not ports:
        fail("No /dev/cu.BioRadio* serial port appeared after re-pairing")
        info("This means macOS didn't recognize the BioRadio as a serial device.")
        info("The RFCOMM/SPP profile negotiation failed at the stack level.")
        return False

    cu_port = ports[0]
    ok(f"Serial port appeared: {cu_port}")

    # Check IOKit state immediately
    rc, out, _ = run(["ioreg", "-l", "-w", "0"], timeout=15)

    bioradio_addr_nocolon = BIORADIO_ADDR.replace(":", "").lower()
    lines = out.split('\n')

    acl_size = None
    for i, line in enumerate(lines):
        if bioradio_addr_nocolon in line.lower():
            start = max(0, i - 5)
            end = min(len(lines), i + 30)
            for j in range(start, end):
                if "MaxACLPacketSize" in lines[j]:
                    val = lines[j].split("=")[-1].strip()
                    acl_size = val
                    if val == "0":
                        fail(f"MaxACLPacketSize = 0 — ACL data path STILL not established")
                    else:
                        ok(f"MaxACLPacketSize = {val} — ACL data path active!")
                    break

    # Check encryption via system_profiler
    rc, out, _ = run(["system_profiler", "SPBluetoothDataType"], timeout=15)
    in_bio = False
    for line in out.split('\n'):
        lower = line.lower()
        if any(h in lower for h in BIORADIO_NAME_HINTS):
            in_bio = True
        elif in_bio:
            if "link level encryption" in lower:
                val = line.split(":")[-1].strip()
                if val == "0":
                    fail(f"Link Level Encryption = 0 — encryption STILL not enabled")
                else:
                    ok(f"Link Level Encryption = {val}")
            elif "connected:" in lower:
                if "yes" in lower:
                    ok("Connected: Yes")
                else:
                    warn("Connected: No")
            elif line.strip() and not line.strip().startswith((" ", "\t")):
                in_bio = False

    # Try serial communication
    info(f"Testing serial communication on {cu_port}...")
    try:
        import serial
        ser = serial.Serial(cu_port, baudrate=460800, timeout=3.0,
                            write_timeout=2.0, rtscts=False, dsrdtr=False)
        ser.dtr = False
        ser.rts = False
        time.sleep(1.0)

        # Drain
        if ser.in_waiting:
            stale = ser.read(ser.in_waiting)
            info(f"Drained {len(stale)} stale bytes")

        # Send firmware query
        ser.write(CMD_GET_FIRMWARE)
        ser.flush()

        response = bytearray()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if ser.in_waiting > 0:
                response.extend(ser.read(ser.in_waiting))
                if len(response) >= 3:
                    break
            time.sleep(0.05)

        ser.close()

        if response:
            ok(f"*** SERIAL RESPONSE: {response.hex(' ')} ***")
            ok("BioRadio is responding! Connection is working!")
            return True
        else:
            fail("No serial response — RFCOMM data path still broken")
            info("")
            info("Despite re-pairing, the RFCOMM session is not established.")
            info("macOS creates the serial port from PersistentPorts config but")
            info("the actual RFCOMM multiplexing layer is never set up.")
            return False

    except ImportError:
        warn("pyserial not available — cannot test serial communication")
        return False
    except Exception as e:
        fail(f"Serial test error: {e}")
        return False


# ---------------------------------------------------------------------------
# Phase 4: HCI-level encryption attempt
# ---------------------------------------------------------------------------
def try_hci_encryption():
    """
    Attempt to enable link-level encryption via IOBluetooth API.
    This is a last-resort attempt using every available method.
    """
    section("PHASE 4: HCI-Level Encryption Attempt")

    try:
        import objc
    except ImportError:
        fail("pyobjc not installed — cannot attempt HCI encryption fix")
        info("Install with: pip install pyobjc-framework-IOBluetooth")
        return False

    info("Loading IOBluetooth framework...")

    try:
        # Load the IOBluetooth framework
        bundle = objc.loadBundle(
            "IOBluetooth",
            bundle_path="/System/Library/Frameworks/IOBluetooth.framework",
            module_globals={},
        )
        IOBluetoothDevice = objc.lookUpClass('IOBluetoothDevice')
    except Exception as e:
        fail(f"Cannot load IOBluetooth: {e}")
        return False

    # Get the device
    device = IOBluetoothDevice.withAddressString_(BIORADIO_ADDR)
    if device is None:
        # Try dash-separated format
        device = IOBluetoothDevice.withAddressString_(BIORADIO_ADDR.replace(":", "-"))
    if device is None:
        fail(f"Cannot find device {BIORADIO_ADDR}")
        return False

    info(f"Device: {device.name()} ({device.addressString()})")
    info(f"Connected: {device.isConnected()}")

    # Check current encryption
    try:
        enc = device.linkLevelEncryption()
        info(f"Link Level Encryption: {enc}")
    except Exception as e:
        warn(f"Cannot read linkLevelEncryption: {e}")

    if not device.isConnected():
        info("Opening connection...")
        result = device.openConnection()
        info(f"openConnection result: {result}")
        time.sleep(2)

    # Method 1: requestAuthentication
    info("Attempting requestAuthentication...")
    try:
        result = device.requestAuthentication()
        info(f"  requestAuthentication result: {result}")
        time.sleep(2)
    except Exception as e:
        warn(f"  requestAuthentication error: {e}")

    # Check encryption after auth
    try:
        enc = device.linkLevelEncryption()
        info(f"  Link Level Encryption after auth: {enc}")
        if enc != 0:
            ok("Encryption enabled after authentication!")
    except Exception:
        pass

    # Method 2: requiresAuthenticationEncryption_
    info("Attempting requiresAuthenticationEncryption_(True)...")
    try:
        result = device.requiresAuthenticationEncryption_(True)
        info(f"  result: {result}")
        time.sleep(2)
    except Exception as e:
        warn(f"  error: {e}")

    # Method 3: openConnection with auth required
    info("Attempting openConnection with authenticationRequired=True...")
    try:
        result = device.openConnection_withPageTimeout_authenticationRequired_(
            None, 15000, True
        )
        info(f"  result: {result}")
        time.sleep(3)
    except Exception as e:
        warn(f"  error: {e}")

    # Check final state
    try:
        enc = device.linkLevelEncryption()
        conn = device.isConnected()
        info(f"Final state — Connected: {conn}, Encryption: {enc}")

        if enc != 0 and conn:
            ok("Encryption is now enabled!")
            return True
        else:
            fail("Could not enable encryption via IOBluetooth API")
            return False
    except Exception as e:
        fail(f"Cannot check final state: {e}")
        return False


# ---------------------------------------------------------------------------
# Phase 5: Verdict and recommendations
# ---------------------------------------------------------------------------
def print_verdict(serial_works):
    """Print final verdict and next steps."""
    section("VERDICT")

    if serial_works:
        print("""
  ╔═══════════════════════════════════════════════════════════╗
  ║  SUCCESS — BioRadio serial communication is working!     ║
  ╚═══════════════════════════════════════════════════════════╝

  You can now use bioradio.py to connect:

    from src.bioradio import BioRadio
    radio = BioRadio()  # or BioRadio(port="/dev/cu.BioRadioAYA")
    radio.connect()
""")
    else:
        print("""
  ╔═══════════════════════════════════════════════════════════╗
  ║  macOS cannot establish RFCOMM data channel to BioRadio  ║
  ╚═══════════════════════════════════════════════════════════╝

  Root cause: macOS Sonoma's Bluetooth stack creates a phantom serial
  port but never completes RFCOMM session setup with this BioRadio.

  RECOMMENDED SOLUTIONS (in order of preference):

  ┌─────────────────────────────────────────────────────────────┐
  │ Option 1: Windows Machine → LSL → Mac (Network Bridge)     │
  │                                                              │
  │ Use a Windows machine to connect to the BioRadio, then      │
  │ stream data to the Mac over the network using Lab Streaming  │
  │ Layer (LSL). Both machines must be on the same network.      │
  │                                                              │
  │ Run this to generate the bridge scripts:                     │
  │   python bioradio_lsl_bridge.py --generate                   │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │ Option 2: Windows VM with USB Bluetooth Adapter             │
  │                                                              │
  │ Run Windows in a VM (Parallels/VMware/VirtualBox), pass a   │
  │ USB Bluetooth adapter through to the VM, and connect the    │
  │ BioRadio from Windows inside the VM. The Mac's built-in     │
  │ Bluetooth won't work — the VM needs its own USB adapter.    │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │ Option 3: Raspberry Pi Bridge                                │
  │                                                              │
  │ A Raspberry Pi (Linux) can connect to the BioRadio via       │
  │ Bluetooth SPP with rfcomm, then forward data over the       │
  │ network. Linux BT Classic support is more reliable for SPP.  │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │ Option 4: Try a different macOS version                      │
  │                                                              │
  │ macOS Ventura (13) or older may handle SPP differently.      │
  │ If you have a Mac on an older version, try pairing there.    │
  └─────────────────────────────────────────────────────────────┘
""")


def _update_address(addr):
    global BIORADIO_ADDR
    BIORADIO_ADDR = addr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BioRadio Bluetooth Nuclear Reset & Diagnostic"
    )
    parser.add_argument("--check-only", action="store_true",
                        help="Only check current state, don't reset")
    parser.add_argument("--hci", action="store_true",
                        help="Attempt HCI-level encryption fix")
    parser.add_argument("--address", default=BIORADIO_ADDR,
                        help=f"BioRadio BT address (default: {BIORADIO_ADDR})")
    args = parser.parse_args()

    # Update module-level address if user specified a different one
    _update_address(args.address)

    print("=" * 70)
    print("  BioRadio Bluetooth Nuclear Reset & Re-pair Diagnostic")
    print(f"  Target device: {BIORADIO_ADDR}")
    print(f"  Platform: macOS {platform.mac_ver()[0]}")
    print("=" * 70)

    # Phase 1: Check current state
    current_ok = check_current_state()

    if args.check_only:
        print_verdict(False)  # TODO: detect if serial actually works
        return

    if args.hci:
        try_hci_encryption()
        # Re-check serial after encryption attempt
        ports = sorted(glob_module.glob("/dev/cu.BioRadio*"))
        if ports:
            try:
                import serial
                ser = serial.Serial(ports[0], baudrate=460800, timeout=3.0,
                                    write_timeout=2.0, rtscts=False, dsrdtr=False)
                ser.dtr = False
                ser.rts = False
                time.sleep(1)
                ser.write(CMD_GET_FIRMWARE)
                ser.flush()
                time.sleep(2)
                resp = bytearray()
                if ser.in_waiting > 0:
                    resp = bytearray(ser.read(ser.in_waiting))
                ser.close()
                if resp:
                    ok(f"Serial response after HCI fix: {resp.hex(' ')}")
                    print_verdict(True)
                    return
                else:
                    fail("No serial response after HCI encryption attempt")
            except Exception as e:
                fail(f"Serial test error: {e}")

        print_verdict(False)
        return

    # Phase 2: Nuclear reset
    did_reset = nuclear_reset()

    if did_reset:
        # Phase 3: Post-pair verification
        serial_works = post_pair_verify()

        if not serial_works:
            # Phase 4: Try HCI encryption as last resort
            hci_ok = try_hci_encryption()
            if hci_ok:
                serial_works = post_pair_verify()

        print_verdict(serial_works)
    else:
        print_verdict(False)


if __name__ == "__main__":
    main()
