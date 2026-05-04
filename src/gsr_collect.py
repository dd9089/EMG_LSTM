"""
BioRobotics Lab 3 - GSR Data Collection (Guided Protocol)
==========================================================

Connects directly to the BioRadio, configures channel 1 for GSR mode,
and runs a timed stress/relaxation protocol with condition markers.

Protocol (Part A):
  Phase 1: Baseline rest (2 min)      — Sit quietly, breathe normally
  Phase 2: Deep breathing (2 min)     — Paced breathing: 5s in / 5s out
  Phase 3: Mental arithmetic (2 min)  — Count backwards from 1000 by 7
  Phase 4: Recovery rest (2 min)      — Sit quietly, relax

Output: CSV file with columns [timestamp, gsr_1, condition]
        plus metadata header matching the visualizer convention.

Usage:
    python src/gsr_collect.py                           # Auto-detect BioRadio
    python src/gsr_collect.py --port COM9               # Specify port
    python src/gsr_collect.py --participant P01          # Set participant ID
    python src/gsr_collect.py --port COM9 --participant P01 --output data/my_gsr.csv

Requirements:
    pip install pyserial
"""

import time
import csv
import os
import sys
import argparse
import threading
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bioradio import (
    BioRadio, BioPotentialMode, CouplingType,
)


# ======================================================================
# Protocol Definition
# ======================================================================

PROTOCOL = [
    {
        "name": "baseline",
        "duration_sec": 120,
        "instruction": "Sit quietly. Relax. Breathe normally.\n"
                       "  Keep your hand still on the table.",
    },
    {
        "name": "breathing",
        "duration_sec": 120,
        "instruction": "Follow the breathing prompts below.\n"
                       "  Breathe in for 5 seconds, out for 5 seconds.",
    },
    {
        "name": "arithmetic",
        "duration_sec": 120,
        "instruction": "Count backwards from 1000 by 7.\n"
                       "  Say each number ALOUD as fast as you can.\n"
                       "  (1000, 993, 986, 979, ...)",
    },
    {
        "name": "recovery",
        "duration_sec": 120,
        "instruction": "Sit quietly again. Relax.\n"
                       "  Let your mind wander. Breathe normally.",
    },
]


# ======================================================================
# Display Helpers
# ======================================================================

def clear_line():
    """Move cursor to beginning of line and clear it."""
    print("\r" + " " * 80 + "\r", end="", flush=True)


def format_time(seconds):
    """Format seconds as M:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def breathing_prompt(elapsed, phase_start):
    """Return breathing cue text based on timing (5s in / 5s out)."""
    t = elapsed - phase_start
    cycle = t % 10
    if cycle < 5:
        remaining = 5 - cycle
        return f"  BREATHE IN... ({remaining:.0f}s)"
    else:
        remaining = 10 - cycle
        return f"  BREATHE OUT... ({remaining:.0f}s)"


# ======================================================================
# Data Collection
# ======================================================================

def collect_gsr(port=None, participant="P01", output_file=None, sample_rate=250):
    """
    Run the full guided GSR collection protocol.

    Parameters
    ----------
    port : str or None
        Serial port for BioRadio (None = auto-detect)
    participant : str
        Participant ID for metadata
    output_file : str or None
        Output CSV path (None = auto-generate)
    sample_rate : int
        BioRadio sample rate in Hz
    """
    # Auto-generate output filename if not specified
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("data", exist_ok=True)
        output_file = f"data/{participant}_gsr_guided_{timestamp}.csv"

    total_duration = sum(p["duration_sec"] for p in PROTOCOL)

    print("\n" + "=" * 60)
    print("  BioRadio GSR Data Collection — Guided Protocol")
    print("=" * 60)
    print(f"  Participant: {participant}")
    print(f"  Output:      {output_file}")
    print(f"  Duration:    {format_time(total_duration)} total")
    print()
    print("  Protocol:")
    for i, phase in enumerate(PROTOCOL):
        print(f"    {i+1}. {phase['name']:12s} ({format_time(phase['duration_sec'])})")
    print()

    # --- Connect and configure BioRadio ---
    print("  Connecting to BioRadio...")
    radio = BioRadio(port=port)

    try:
        radio.connect()
        config = radio.get_configuration()
        print(f"  Connected: {radio.device_name}")

        # Configure channel 1 for GSR
        bp_channels = config.biopotential_channels
        if not bp_channels:
            print("  ERROR: No BioPotential channels found!")
            return

        ch = bp_channels[0]
        ch.operation_mode = BioPotentialMode.GSR
        ch.coupling = CouplingType.DC
        ch.bit_resolution = 16
        ch.enabled = True
        ch.name = "GSR"
        radio.set_channel_config(ch)

        # Set sample rate
        if config.sample_rate != sample_rate:
            radio.set_sample_rate(sample_rate)
            config = radio.get_configuration()

        print(f"  GSR mode configured: Ch{ch.channel_index}, "
              f"{config.sample_rate}Hz, DC coupling")

        # Battery check
        battery = radio.get_battery_info()
        print(f"  Battery: {battery.voltage:.2f}V ({battery.percentage:.0f}%)")

        if battery.percentage < 20:
            print("  WARNING: Battery low! Consider charging before recording.")

        print()
        print("  Electrode placement:")
        print("    - Place electrodes on INDEX and MIDDLE fingers")
        print("    - Non-dominant hand")
        print("    - Rest hand comfortably on the table")
        print("    - Do NOT move the hand with electrodes during recording")
        print()

        input("  Press ENTER when ready to begin... ")
        print()

        # --- Acquire data with protocol timing ---
        all_data = []       # (timestamp, value, condition)
        stop_flag = False

        def acquisition_loop():
            """Background thread: read samples from BioRadio."""
            nonlocal stop_flag
            while not stop_flag:
                sample = radio.read_data(timeout=0.1)
                if sample and sample.biopotential:
                    ts = time.time()
                    first_ch = next(iter(sample.biopotential))
                    vals = sample.biopotential[first_ch]
                    for v in vals:
                        all_data.append((ts, v, current_condition[0]))

        current_condition = [""]  # Mutable container for thread access

        radio.start_acquisition()
        acq_thread = threading.Thread(target=acquisition_loop, daemon=True)
        acq_thread.start()

        start_time = time.time()

        try:
            for phase_idx, phase in enumerate(PROTOCOL):
                phase_name = phase["name"]
                phase_duration = phase["duration_sec"]
                current_condition[0] = phase_name

                print(f"  {'=' * 50}")
                print(f"  Phase {phase_idx + 1}/{len(PROTOCOL)}: "
                      f"{phase_name.upper()} ({format_time(phase_duration)})")
                print(f"  {'=' * 50}")
                print(f"  {phase['instruction']}")
                print()

                phase_start = time.time()
                phase_elapsed = 0

                while phase_elapsed < phase_duration:
                    phase_elapsed = time.time() - phase_start
                    total_elapsed = time.time() - start_time
                    remaining = phase_duration - phase_elapsed

                    # Build status line
                    n_samples = len(all_data)
                    status = (f"  [{phase_name}] "
                              f"{format_time(phase_elapsed)}/{format_time(phase_duration)} "
                              f"| Total: {format_time(total_elapsed)} "
                              f"| Samples: {n_samples}")

                    # Add breathing prompts during breathing phase
                    if phase_name == "breathing":
                        breath = breathing_prompt(time.time(), phase_start)
                        status += f"  {breath}"

                    clear_line()
                    print(status, end="", flush=True)
                    time.sleep(0.5)

                print()  # Newline after phase
                print(f"  Phase '{phase_name}' complete.\n")

        except KeyboardInterrupt:
            print("\n\n  Recording interrupted by user.")

        finally:
            stop_flag = True
            acq_thread.join(timeout=2.0)
            radio.stop_acquisition()

        # --- Save data ---
        if not all_data:
            print("  No data collected!")
            return

        print(f"\n  Saving {len(all_data)} samples to {output_file}...")

        # Compute timing stats
        t0 = all_data[0][0]
        t_end = all_data[-1][0]
        actual_duration = t_end - t0
        effective_rate = (len(all_data) - 1) / actual_duration if actual_duration > 0 else 0

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

        with open(output_file, "w", newline="") as f:
            # Metadata header (matches visualizer convention)
            f.write(f"# participant_id: {participant}\n")
            f.write(f"# protocol: guided_stress_relaxation\n")
            f.write(f"# stream_name: BioRadio_GSR\n")
            f.write(f"# stream_type: GSR\n")
            f.write(f"# timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            f.write(f"# samples: {len(all_data)}\n")
            f.write(f"# duration_sec: {actual_duration:.3f}\n")
            f.write(f"# nominal_sample_rate: {sample_rate}\n")
            f.write(f"# effective_sample_rate: {effective_rate:.2f}\n")
            f.write(f"# device: {radio.device_name}\n")
            f.write(f"# firmware: {radio.firmware_version}\n")
            f.write("#\n")

            # Column header
            f.write("timestamp,gsr_1,condition\n")

            # Data rows (timestamp relative to recording start)
            for ts, val, cond in all_data:
                f.write(f"{ts - t0:.6f},{val:.6f},{cond}\n")

        print(f"  Saved: {output_file}")
        print(f"  Duration: {actual_duration:.1f}s")
        print(f"  Effective rate: {effective_rate:.1f} Hz")
        print(f"  Dropped packets: {radio.dropped_packets}")

        # Per-condition summary
        print(f"\n  Per-condition sample counts:")
        from collections import Counter
        counts = Counter(row[2] for row in all_data)
        for cond_name in [p["name"] for p in PROTOCOL]:
            n = counts.get(cond_name, 0)
            print(f"    {cond_name:12s}: {n:6d} samples "
                  f"({n / effective_rate:.1f}s)" if effective_rate > 0 else "")

    finally:
        radio.disconnect()

    print("\n  Done! Open the analysis notebook to process your data.")
    return output_file


# ======================================================================
# CLI Entry Point
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BioRadio GSR Data Collection — Guided Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Protocol:
  Phase 1: Baseline rest      (2 min)
  Phase 2: Deep breathing      (2 min)
  Phase 3: Mental arithmetic   (2 min)
  Phase 4: Recovery rest       (2 min)

Examples:
  python src/gsr_collect.py                             # Auto-detect
  python src/gsr_collect.py --port COM9                 # Specify port
  python src/gsr_collect.py --port COM9 --participant P01
        """
    )
    parser.add_argument("--port", "-p", default=None,
                        help="Serial port (e.g. COM9)")
    parser.add_argument("--participant", default="P01",
                        help="Participant ID (default: P01)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV file path (default: auto-generate)")
    parser.add_argument("--rate", type=int, default=250,
                        help="Sample rate in Hz (default: 250)")
    args = parser.parse_args()

    collect_gsr(
        port=args.port,
        participant=args.participant,
        output_file=args.output,
        sample_rate=args.rate,
    )
