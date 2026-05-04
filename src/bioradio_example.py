"""
bioradio_example.py - Student-friendly examples for using the BioRadio in Python.

This script demonstrates how to:
  1. Scan for the BioRadio serial port (works on Windows AND macOS)
  2. Connect to the device
  3. Read configuration and battery status
  4. Acquire EMG/BioPotential data
  5. Stream data to LSL for visualization
  6. Save data to CSV

No .NET SDK or BioCapture software required - just Python + pyserial!

Usage:
    python src/bioradio_example.py                          # auto-detect
    python src/bioradio_example.py --port COM9              # Windows
    python src/bioradio_example.py --port /dev/cu.BioRadioAYA  # macOS (use cu.* not tty.*!)

Requirements:
    pip install pyserial
    (optional) pip install pylsl  -- for LSL streaming
    (optional) pip install matplotlib  -- for live plot
"""

import time
import csv
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bioradio import (
    BioRadio, scan_for_bioradio, find_bioradio_port,
    probe_bioradio_port, create_lsl_outlet,
)


# =====================================================================
# Example 1: Scan for COM Ports
# =====================================================================
def example_scan():
    """
    Students may not know which COM port their BioRadio is on.
    This scans all ports and probes each one to find the working port.
    """
    print("\n" + "="*60)
    print("  EXAMPLE 1: Scanning for BioRadio ports")
    print("="*60)

    port = find_bioradio_port(verbose=True)

    if port:
        print(f"\n  Use this port:  BioRadio(port='{port}')")
    else:
        print("  No BioRadio ports found!")
        print("  Make sure the device is powered on and paired/plugged in.")

    return port


# =====================================================================
# Example 2: Connect and Read Device Info
# =====================================================================
def example_device_info(port=None):
    """
    Connect to the BioRadio and print device information.
    """
    print("\n" + "="*60)
    print("  EXAMPLE 2: Device Info")
    print("="*60)

    radio = BioRadio(port=port)

    try:
        radio.connect()
        print(f"  Device Name: {radio.device_name}")
        print(f"  Firmware:    {radio.firmware_version}")
        print(f"  Hardware:    {radio.hardware_version}")

        # Read full configuration
        config = radio.get_configuration()
        print(f"\n  Sample Rate: {config.sample_rate} Hz")
        print(f"  Termination: {'Single-Ended (8ch)' if config.is_single_ended else 'Differential (4ch)'}")

        # Battery
        battery = radio.get_battery_info()
        print(f"  Battery:     {battery.voltage:.2f}V ({battery.percentage:.0f}%)")

        # Channel list
        print(f"\n  Active Channels:")
        for ch in config.enabled_biopotential:
            print(f"    Ch{ch.channel_index}: {ch.name or 'unnamed'} "
                  f"({ch.bit_resolution}bit, gain={ch.gain}, {ch.coupling.name})")

        if config.enabled_auxiliary:
            print(f"  Auxiliary Channels:")
            for ch in config.enabled_auxiliary:
                print(f"    Ch{ch.channel_index}: {ch.name or 'unnamed'}")

        if config.enabled_pulseox:
            print(f"  Pulse Ox Channels:")
            for ch in config.enabled_pulseox:
                print(f"    Ch{ch.channel_index}: {ch.name or 'unnamed'}")

    finally:
        radio.disconnect()

    return config


# =====================================================================
# Example 3: Acquire Data for N Seconds
# =====================================================================
def example_acquire(port=None, duration=5.0):
    """
    Acquire BioPotential data for a specified duration and print stats.
    """
    print("\n" + "="*60)
    print(f"  EXAMPLE 3: Acquire Data ({duration}s)")
    print("="*60)

    radio = BioRadio(port=port)
    all_samples = []

    try:
        radio.connect()
        config = radio.get_configuration()
        radio.start_acquisition()

        print(f"  Acquiring at {config.sample_rate}Hz...")
        start = time.time()

        while time.time() - start < duration:
            sample = radio.read_data(timeout=0.1)
            if sample:
                all_samples.append(sample)

                # Print progress every ~1 second
                if len(all_samples) % 125 == 0:
                    elapsed = time.time() - start
                    print(f"  {elapsed:.1f}s: {len(all_samples)} packets, "
                          f"dropped={radio.dropped_packets}")

        radio.stop_acquisition()

    finally:
        radio.disconnect()

    print(f"\n  Collected {len(all_samples)} data packets")
    print(f"  Dropped: {radio.dropped_packets}")

    return all_samples


# =====================================================================
# Example 4: Save Data to CSV
# =====================================================================
def example_save_csv(port=None, duration=5.0, filename="bioradio_data.csv"):
    """
    Acquire data and save all BioPotential channels to a CSV file.
    """
    print("\n" + "="*60)
    print(f"  EXAMPLE 4: Save to CSV ({filename})")
    print("="*60)

    radio = BioRadio(port=port)

    try:
        radio.connect()
        config = radio.get_configuration()

        bp_channels = config.enabled_biopotential
        if not bp_channels:
            print("  No enabled BioPotential channels!")
            return

        # Prepare CSV
        filepath = os.path.join("data", filename)
        os.makedirs("data", exist_ok=True)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            header = ["timestamp", "packet_id"]
            for ch in bp_channels:
                header.append(f"Ch{ch.channel_index}_{ch.name or 'BP'}")
            writer.writerow(header)

            # Acquire
            radio.start_acquisition()
            print(f"  Recording {len(bp_channels)} channels for {duration}s...")
            start = time.time()
            row_count = 0

            while time.time() - start < duration:
                sample = radio.read_data(timeout=0.1)
                if sample and sample.biopotential:
                    # Each sample may contain multiple sub-samples per channel
                    num_subsamples = max(
                        len(v) for v in sample.biopotential.values()
                    ) if sample.biopotential else 0

                    for s_idx in range(num_subsamples):
                        row = [sample.timestamp, sample.packet_id]
                        for ch in bp_channels:
                            vals = sample.biopotential.get(ch.channel_index, [])
                            row.append(vals[s_idx] if s_idx < len(vals) else "")
                        writer.writerow(row)
                        row_count += 1

            radio.stop_acquisition()

        print(f"  Saved {row_count} samples to {filepath}")

    finally:
        radio.disconnect()


# =====================================================================
# Example 5: Real-time Callback
# =====================================================================
def example_callback(port=None, duration=3.0):
    """
    Use a callback function to process each data packet in real-time.
    This is useful for real-time control applications.
    """
    print("\n" + "="*60)
    print(f"  EXAMPLE 5: Real-time Callback ({duration}s)")
    print("="*60)

    packet_count = [0]  # Using list to allow mutation in closure

    def my_callback(sample):
        """Called for every data packet received from the BioRadio."""
        packet_count[0] += 1
        if packet_count[0] % 125 == 0:  # Print every ~1s
            if sample.biopotential:
                first_ch_idx = next(iter(sample.biopotential))
                vals = sample.biopotential[first_ch_idx]
                print(f"  Packet #{sample.packet_id}: "
                      f"Ch{first_ch_idx} = {vals[:3]}... "
                      f"Battery={sample.battery_voltage:.2f}V")

    radio = BioRadio(port=port)

    try:
        radio.connect()
        radio.get_configuration()

        # Register callback
        radio.on_data(my_callback)

        radio.start_acquisition()
        print(f"  Listening for {duration}s...")
        time.sleep(duration)
        radio.stop_acquisition()

    finally:
        radio.disconnect()

    print(f"  Received {packet_count[0]} packets via callback")


# =====================================================================
# Example 6: Stream to LSL
# =====================================================================
def example_lsl_stream(port=None, duration=10.0):
    """
    Stream BioRadio data to the Lab Streaming Layer.
    Open the visualizer (src/visualizer.py) to see the data!
    """
    print("\n" + "="*60)
    print(f"  EXAMPLE 6: LSL Streaming ({duration}s)")
    print("="*60)

    try:
        import pylsl
    except ImportError:
        print("  pylsl not installed! Run: pip install pylsl")
        return

    radio = BioRadio(port=port)

    try:
        radio.connect()
        config = radio.get_configuration()

        # Create LSL outlet
        outlet = create_lsl_outlet(config)
        if not outlet:
            print("  Could not create LSL outlet (no enabled channels?)")
            return

        bp_channels = config.enabled_biopotential
        print(f"  LSL stream: 'BioRadio_{config.name}' "
              f"({len(bp_channels)}ch @ {config.sample_rate}Hz)")
        print(f"  Open src/visualizer.py to see the data!")

        radio.start_acquisition()
        start = time.time()
        pushed = 0

        while time.time() - start < duration:
            sample = radio.read_data(timeout=0.1)
            if sample and sample.biopotential:
                num_sub = max(len(v) for v in sample.biopotential.values())
                for s_idx in range(num_sub):
                    lsl_sample = []
                    for ch in bp_channels:
                        vals = sample.biopotential.get(ch.channel_index, [])
                        lsl_sample.append(
                            float(vals[s_idx]) if s_idx < len(vals) else 0.0
                        )
                    outlet.push_sample(lsl_sample)
                    pushed += 1

            if pushed % 1000 == 0 and pushed > 0:
                print(f"  Pushed {pushed} samples to LSL...")

        radio.stop_acquisition()
        print(f"  Total: {pushed} samples pushed to LSL")

    finally:
        radio.disconnect()


# =====================================================================
# Example 7: Configure Channel for GSR Mode
# =====================================================================
def example_gsr_config(port=None, duration=10.0):
    """
    Configure a BioRadio channel for Galvanic Skin Response (GSR) recording.

    GSR measures electrodermal activity (EDA) — tiny changes in skin
    conductance caused by sweat gland activity, controlled by the
    sympathetic nervous system.

    GSR requires:
    - BioPotentialMode.GSR (operation_mode = 1)
    - DC coupling (CouplingType.DC) — GSR is a slowly varying DC signal
    - Lower sample rate is fine (250 Hz) since GSR is < 5 Hz
    - Finger electrodes on index and middle fingers (non-dominant hand)
    """
    print("\n" + "="*60)
    print(f"  EXAMPLE 7: GSR Configuration & Acquisition ({duration}s)")
    print("="*60)

    from src.bioradio import (
        BioRadio, BioPotentialMode, CouplingType,
        create_lsl_outlet,
    )

    radio = BioRadio(port=port)

    try:
        radio.connect()
        config = radio.get_configuration()

        # --- Configure channel 1 for GSR ---
        bp_channels = config.biopotential_channels
        if not bp_channels:
            print("  No BioPotential channels found on device!")
            return

        ch = bp_channels[0]
        print(f"\n  Before: Ch{ch.channel_index} mode={ch.operation_mode.name}, "
              f"coupling={ch.coupling.name}")

        ch.operation_mode = BioPotentialMode.GSR
        ch.coupling = CouplingType.DC
        ch.bit_resolution = 16
        ch.enabled = True
        ch.name = "GSR"

        radio.set_channel_config(ch)

        # Set sample rate to 250 Hz (sufficient for GSR)
        if config.sample_rate != 250:
            print(f"  Setting sample rate: {config.sample_rate} Hz -> 250 Hz ...")
            radio.set_sample_rate(250)

        # Re-read config to confirm
        config = radio.get_configuration()
        print(f"\n  After configuration:")
        for c in config.enabled_biopotential:
            print(f"    Ch{c.channel_index}: '{c.name}' mode={c.operation_mode.name}, "
                  f"coupling={c.coupling.name}, {c.bit_resolution}bit")
        print(f"  Sample rate: {config.sample_rate} Hz")

        # --- Acquire GSR data ---
        radio.start_acquisition()
        print(f"\n  Recording GSR for {duration}s...")
        print("  (Place finger electrodes on index + middle fingers)\n")

        import time as _time
        start = _time.time()
        all_values = []

        while _time.time() - start < duration:
            sample = radio.read_data(timeout=0.1)
            if sample and sample.biopotential:
                first_ch = next(iter(sample.biopotential))
                vals = sample.biopotential[first_ch]
                all_values.extend(vals)

                if len(all_values) % 250 == 0:
                    recent = all_values[-250:]
                    mean_val = sum(recent) / len(recent)
                    print(f"  {_time.time()-start:.1f}s: "
                          f"GSR = {mean_val:.4f} (mean of last 1s)")

        radio.stop_acquisition()

        print(f"\n  Collected {len(all_values)} GSR samples")
        if all_values:
            import statistics
            print(f"  Mean:  {statistics.mean(all_values):.4f}")
            print(f"  Std:   {statistics.stdev(all_values):.4f}")
            print(f"  Range: {min(all_values):.4f} — {max(all_values):.4f}")

    finally:
        radio.disconnect()


# =====================================================================
# Main: run all examples interactively
# =====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BioRadio Examples")
    parser.add_argument("--port", "-p", default=None,
                        help="Serial port (e.g. COM9 or /dev/cu.BioRadioAYA)")
    # Legacy arguments (hidden, backwards compat)
    parser.add_argument("--in", dest="port_in", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--out", dest="port_out", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--example", type=int, default=0,
                        choices=[0, 1, 2, 3, 4, 5, 6, 7],
                        help="Run specific example (0=menu)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Duration for acquisition examples (seconds)")
    args = parser.parse_args()

    # Resolve port: explicit --port, or legacy --in, or auto-detect
    port = args.port or args.port_in
    # port=None means auto-detect (BioRadio.connect() handles it)

    examples = {
        1: ("Scan for ports",       lambda: example_scan()),
        2: ("Device info",          lambda: example_device_info(port)),
        3: ("Acquire data",         lambda: example_acquire(port, args.duration)),
        4: ("Save to CSV",          lambda: example_save_csv(port, args.duration)),
        5: ("Real-time callback",   lambda: example_callback(port, args.duration)),
        6: ("LSL streaming",        lambda: example_lsl_stream(port, args.duration)),
        7: ("GSR config + acquire", lambda: example_gsr_config(port, args.duration)),
    }

    if args.example > 0:
        examples[args.example][1]()
    else:
        print("\n" + "="*60)
        print("  BioRadio Python Examples")
        print("="*60)
        print(f"  Port: {port or '(auto-detect)'}\n")
        for num, (name, _) in examples.items():
            print(f"  [{num}] {name}")
        print(f"  [0] Exit\n")

        while True:
            try:
                choice = input("  Select example (1-7, 0 to exit): ").strip()
                if choice == "0" or choice == "":
                    break
                num = int(choice)
                if num in examples:
                    examples[num][1]()
                else:
                    print("  Invalid choice!")
            except KeyboardInterrupt:
                break
            except ValueError:
                print("  Enter a number!")

    print("\nDone!")
