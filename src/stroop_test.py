"""
BioRobotics Lab 3 - Stroop Color-Word Test (PyQt6 GUI)
========================================================

A full-screen GUI Stroop test for GSR/EDA experiments.

The Stroop effect: naming the INK COLOR of a word is slower and harder
when the word spells a different color (incongruent) vs. the same color
(congruent). This cognitive interference activates the sympathetic nervous
system, producing measurable GSR responses.

Features:
  - Settings dialog to configure: trials, congruent ratio, timing
  - Full-screen stimulus presentation with large colored text
  - Keyboard-based responses (R/B/G/Y) for precise reaction time
  - High-resolution timing via time.perf_counter()
  - CSV output with per-trial data + parameter metadata
  - Real-time accuracy feedback

Students adjust parameters to test hypotheses about task difficulty
and its effect on GSR (e.g., shorter display time = harder = more stress).

Usage:
    python src/stroop_test.py                    # Launch with settings dialog
    python src/stroop_test.py --trials 40        # Override default trial count
    python src/stroop_test.py --output results.csv

Requirements:
    PyQt6 (already in environment.yml)

Author: BioRobotics Course
"""

import time
import csv
import os
import sys
import random
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QDialog, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QPushButton, QSpinBox,
    QGroupBox, QSizePolicy, QLineEdit, QComboBox,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QKeyEvent, QColor


# ======================================================================
# Constants
# ======================================================================

COLOR_MAP = {
    'RED':    '#FF3333',
    'BLUE':   '#3399FF',
    'GREEN':  '#33CC33',
    'YELLOW': '#FFCC00',
}

RESPONSE_KEYS = {
    Qt.Key.Key_R: 'RED',
    Qt.Key.Key_B: 'BLUE',
    Qt.Key.Key_G: 'GREEN',
    Qt.Key.Key_Y: 'YELLOW',
}

COLOR_NAMES = list(COLOR_MAP.keys())

BACKGROUND_COLOR = '#1a1a2e'
TEXT_COLOR = '#e0e0e0'
CORRECT_COLOR = '#33CC33'
INCORRECT_COLOR = '#FF3333'


# ======================================================================
# Data Structures
# ======================================================================

@dataclass
class StroopParams:
    """Configurable Stroop test parameters."""
    participant_id: str = "P00"
    condition: str = "baseline"
    n_trials: int = 30
    congruent_pct: int = 50
    stimulus_time_ms: int = 3000
    iti_ms: int = 2000


@dataclass
class Trial:
    """A single Stroop trial."""
    number: int
    word: str
    ink_color: str
    congruent: bool
    response_key: str = ""
    correct: bool = False
    response_time_ms: float = 0.0
    timestamp_sec: float = 0.0
    stimulus_datetime: str = ""


# ======================================================================
# Settings Dialog
# ======================================================================

class StroopSettingsDialog(QDialog):
    """Pre-test settings dialog for parameter configuration."""

    def __init__(self, params: StroopParams, parent=None):
        super().__init__(parent)
        self.params = params
        self.setWindowTitle("Stroop Test Settings")
        self.setMinimumWidth(500)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Stroop Color-Word Test")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel(
            "Adjust parameters below, then click Start.\n"
            "Task: Press the key matching the INK COLOR (not the word).\n"
            "Keys: R = Red, B = Blue, G = Green, Y = Yellow"
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(10)

        # Participant info group
        info_group = QGroupBox("Participant Info")
        info_layout = QGridLayout(info_group)

        info_layout.addWidget(QLabel("Participant ID:"), 0, 0)
        self.participant_id_edit = QLineEdit(self.params.participant_id)
        self.participant_id_edit.setPlaceholderText("e.g., P01")
        info_layout.addWidget(self.participant_id_edit, 0, 1)

        info_layout.addWidget(QLabel("Condition:"), 1, 0)
        self.condition_combo = QComboBox()
        self.condition_combo.setEditable(True)
        self.condition_combo.addItems([
            "practice", "low stress", "high stress",
        ])
        idx = self.condition_combo.findText(self.params.condition)
        if idx >= 0:
            self.condition_combo.setCurrentIndex(idx)
        else:
            self.condition_combo.setCurrentText(self.params.condition)
        info_layout.addWidget(self.condition_combo, 1, 1)

        layout.addWidget(info_group)

        layout.addSpacing(10)

        # Parameters group
        params_group = QGroupBox("Parameters")
        params_layout = QGridLayout(params_group)

        # --- Number of trials ---
        params_layout.addWidget(QLabel("Number of Trials:"), 0, 0)
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(10, 100)
        self.trials_spin.setValue(self.params.n_trials)
        self.trials_spin.setSingleStep(5)
        params_layout.addWidget(self.trials_spin, 0, 1)
        self.trials_label = QLabel(f"{self.params.n_trials}")
        params_layout.addWidget(self.trials_label, 0, 2)
        self.trials_spin.valueChanged.connect(
            lambda v: self.trials_label.setText(str(v)))

        # --- Congruent ratio ---
        params_layout.addWidget(QLabel("Congruent Ratio:"), 1, 0)
        self.congruent_slider = QSlider(Qt.Orientation.Horizontal)
        self.congruent_slider.setRange(0, 100)
        self.congruent_slider.setValue(self.params.congruent_pct)
        self.congruent_slider.setTickInterval(10)
        self.congruent_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        params_layout.addWidget(self.congruent_slider, 1, 1)
        self.congruent_label = QLabel(f"{self.params.congruent_pct}%")
        params_layout.addWidget(self.congruent_label, 1, 2)
        self.congruent_slider.valueChanged.connect(
            lambda v: self.congruent_label.setText(f"{v}%"))

        # --- Stimulus display time ---
        params_layout.addWidget(QLabel("Stimulus Time:"), 2, 0)
        self.stimulus_slider = QSlider(Qt.Orientation.Horizontal)
        self.stimulus_slider.setRange(500, 5000)
        self.stimulus_slider.setValue(self.params.stimulus_time_ms)
        self.stimulus_slider.setSingleStep(250)
        self.stimulus_slider.setTickInterval(500)
        self.stimulus_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        params_layout.addWidget(self.stimulus_slider, 2, 1)
        self.stimulus_label = QLabel(f"{self.params.stimulus_time_ms / 1000:.1f}s")
        params_layout.addWidget(self.stimulus_label, 2, 2)
        self.stimulus_slider.valueChanged.connect(
            lambda v: self.stimulus_label.setText(f"{v / 1000:.1f}s"))

        # --- Inter-trial interval ---
        params_layout.addWidget(QLabel("Inter-Trial Interval:"), 3, 0)
        self.iti_slider = QSlider(Qt.Orientation.Horizontal)
        self.iti_slider.setRange(500, 4000)
        self.iti_slider.setValue(self.params.iti_ms)
        self.iti_slider.setSingleStep(250)
        self.iti_slider.setTickInterval(500)
        self.iti_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        params_layout.addWidget(self.iti_slider, 3, 1)
        self.iti_label = QLabel(f"{self.params.iti_ms / 1000:.1f}s")
        params_layout.addWidget(self.iti_label, 3, 2)
        self.iti_slider.valueChanged.connect(
            lambda v: self.iti_label.setText(f"{v / 1000:.1f}s"))

        layout.addWidget(params_group)

        # Hypothesis prompt
        hyp_label = QLabel(
            "Before starting, write your hypothesis:\n"
            "How do you expect these parameters to affect GSR?\n"
            "(e.g., \"Shorter stimulus time will increase GSR because...\")"
        )
        hyp_label.setStyleSheet("color: #FFD700; font-style: italic;")
        hyp_label.setWordWrap(True)
        hyp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hyp_label)

        layout.addSpacing(10)

        # Buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Test")
        self.start_btn.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.start_btn.setMinimumHeight(50)
        self.start_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

    def get_params(self) -> StroopParams:
        """Return the configured parameters."""
        return StroopParams(
            participant_id=self.participant_id_edit.text().strip() or "P00",
            condition=self.condition_combo.currentText().strip() or "baseline",
            n_trials=self.trials_spin.value(),
            congruent_pct=self.congruent_slider.value(),
            stimulus_time_ms=self.stimulus_slider.value(),
            iti_ms=self.iti_slider.value(),
        )


# ======================================================================
# Stroop Test Window
# ======================================================================

class StroopTestWindow(QMainWindow):
    """Full-screen Stroop test presentation window."""

    def __init__(self, params: StroopParams, output_file: str, parent=None):
        super().__init__(parent)
        self.params = params
        self.output_file = output_file
        self.trials: List[Trial] = []
        self.results: List[Trial] = []
        self.current_trial_idx = -1
        self.current_trial: Optional[Trial] = None
        self.test_start_time = 0.0
        self.stimulus_onset_time = 0.0
        self.stimulus_onset_datetime = ""
        self.awaiting_response = False
        self.test_finished = False

        # Cancellable timeout timer — prevents stale timeouts from
        # firing into later trials when the participant responds quickly.
        self.timeout_timer = QTimer(self)
        self.timeout_timer.setSingleShot(True)
        self.timeout_timer.timeout.connect(self.stimulus_timeout)

        self.setup_ui()
        self.generate_trials()

    def setup_ui(self):
        self.setWindowTitle("Stroop Test")
        self.setStyleSheet(f"background-color: {BACKGROUND_COLOR};")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        # Progress label (top)
        self.progress_label = QLabel("")
        self.progress_label.setFont(QFont("Arial", 14))
        self.progress_label.setStyleSheet(f"color: {TEXT_COLOR}; padding: 10px;")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_label)

        # Spacer
        layout.addStretch(1)

        # Stimulus label (center — large text)
        self.stimulus_label = QLabel("")
        self.stimulus_label.setFont(QFont("Arial", 120, QFont.Weight.Bold))
        self.stimulus_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stimulus_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.stimulus_label)

        # Feedback label (below stimulus)
        self.feedback_label = QLabel("")
        self.feedback_label.setFont(QFont("Arial", 24))
        self.feedback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.feedback_label)

        # Spacer
        layout.addStretch(1)

        # Response key hints (bottom)
        keys_text = "  |  ".join(
            f"{key}: {color}" for key, color in
            [("R", "Red"), ("B", "Blue"), ("G", "Green"), ("Y", "Yellow")]
        )
        self.keys_label = QLabel(keys_text)
        self.keys_label.setFont(QFont("Courier", 16))
        self.keys_label.setStyleSheet(f"color: {TEXT_COLOR}; padding: 20px;")
        self.keys_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.keys_label)

    def generate_trials(self):
        """Generate randomized trial list based on parameters."""
        n = self.params.n_trials
        n_congruent = int(n * self.params.congruent_pct / 100)
        n_incongruent = n - n_congruent

        trials = []

        # Congruent trials
        for i in range(n_congruent):
            color = random.choice(COLOR_NAMES)
            trials.append(Trial(
                number=0,  # assigned after shuffle
                word=color,
                ink_color=color,
                congruent=True,
            ))

        # Incongruent trials
        for i in range(n_incongruent):
            word = random.choice(COLOR_NAMES)
            ink = random.choice([c for c in COLOR_NAMES if c != word])
            trials.append(Trial(
                number=0,
                word=word,
                ink_color=ink,
                congruent=False,
            ))

        random.shuffle(trials)

        # Assign trial numbers
        for i, trial in enumerate(trials):
            trial.number = i + 1

        self.trials = trials

    def start_test(self):
        """Begin the test sequence."""
        self.test_start_time = time.perf_counter()
        self.stimulus_label.setText("Get Ready...")
        self.stimulus_label.setStyleSheet(f"color: {TEXT_COLOR};")
        self.progress_label.setText(
            f"Test starting... ({self.params.n_trials} trials)")

        # Wait 2 seconds then show first trial
        QTimer.singleShot(2000, self.next_trial)

    def next_trial(self):
        """Advance to the next trial."""
        self.current_trial_idx += 1

        if self.current_trial_idx >= len(self.trials):
            self.finish_test()
            return

        self.current_trial = self.trials[self.current_trial_idx]
        self.progress_label.setText(
            f"Trial {self.current_trial.number} / {len(self.trials)}")

        # Show fixation cross during ITI
        self.stimulus_label.setText("+")
        self.stimulus_label.setStyleSheet(f"color: {TEXT_COLOR};")
        self.feedback_label.setText("")
        self.awaiting_response = False

        # After ITI, show stimulus
        QTimer.singleShot(self.params.iti_ms, self.show_stimulus)

    def show_stimulus(self):
        """Display the color word stimulus."""
        if self.current_trial is None:
            return

        trial = self.current_trial
        hex_color = COLOR_MAP[trial.ink_color]

        self.stimulus_label.setText(trial.word)
        self.stimulus_label.setStyleSheet(f"color: {hex_color};")

        self.stimulus_onset_time = time.perf_counter()
        self.stimulus_onset_datetime = datetime.now().isoformat(timespec='milliseconds')
        self.awaiting_response = True

        # Timeout: if no response within stimulus time, record as timeout
        self.timeout_timer.start(self.params.stimulus_time_ms)

    def stimulus_timeout(self):
        """Handle stimulus timeout (no response given)."""
        if not self.awaiting_response:
            return  # Already responded

        trial = self.current_trial
        if trial is None:
            return

        trial.response_key = "TIMEOUT"
        trial.correct = False
        trial.response_time_ms = self.params.stimulus_time_ms
        trial.timestamp_sec = self.stimulus_onset_time - self.test_start_time
        trial.stimulus_datetime = self.stimulus_onset_datetime

        self.awaiting_response = False
        self.results.append(trial)

        self.feedback_label.setText("TOO SLOW")
        self.feedback_label.setStyleSheet(f"color: {INCORRECT_COLOR};")

        QTimer.singleShot(500, self.next_trial)

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard responses during test."""
        key = event.key()

        # Escape to quit — if test is done, close the window
        if key == Qt.Key.Key_Escape:
            if self.test_finished:
                self.close()
            else:
                self.finish_test()
            return

        # Only process during stimulus
        if not self.awaiting_response:
            return

        if key not in RESPONSE_KEYS:
            return  # Ignore non-response keys

        response_time = time.perf_counter()
        trial = self.current_trial
        if trial is None:
            return

        # Record response
        response_color = RESPONSE_KEYS[key]
        rt_ms = (response_time - self.stimulus_onset_time) * 1000

        # Cancel the timeout timer so it doesn't fire into a later trial
        self.timeout_timer.stop()

        trial.response_key = response_color
        trial.correct = (response_color == trial.ink_color)
        trial.response_time_ms = round(rt_ms, 1)
        trial.timestamp_sec = round(
            self.stimulus_onset_time - self.test_start_time, 3)
        trial.stimulus_datetime = self.stimulus_onset_datetime

        self.awaiting_response = False
        self.results.append(trial)

        # Show feedback
        if trial.correct:
            self.feedback_label.setText("Correct")
            self.feedback_label.setStyleSheet(f"color: {CORRECT_COLOR};")
        else:
            self.feedback_label.setText(f"Incorrect (was {trial.ink_color})")
            self.feedback_label.setStyleSheet(f"color: {INCORRECT_COLOR};")

        QTimer.singleShot(500, self.next_trial)

    def finish_test(self):
        """End the test and save results."""
        if self.test_finished:
            return
        self.test_finished = True
        self.awaiting_response = False

        # Show summary
        if self.results:
            correct = sum(1 for t in self.results if t.correct)
            total = len(self.results)
            accuracy = correct / total * 100

            congruent_rts = [t.response_time_ms for t in self.results
                            if t.congruent and t.correct and t.response_key != "TIMEOUT"]
            incongruent_rts = [t.response_time_ms for t in self.results
                               if not t.congruent and t.correct and t.response_key != "TIMEOUT"]

            summary_parts = [
                f"Test Complete!",
                f"Accuracy: {correct}/{total} ({accuracy:.0f}%)",
            ]
            if congruent_rts:
                summary_parts.append(
                    f"Congruent RT: {sum(congruent_rts)/len(congruent_rts):.0f} ms")
            if incongruent_rts:
                summary_parts.append(
                    f"Incongruent RT: {sum(incongruent_rts)/len(incongruent_rts):.0f} ms")

            self.stimulus_label.setFont(QFont("Arial", 36, QFont.Weight.Bold))
            self.stimulus_label.setText("\n".join(summary_parts))
            self.stimulus_label.setStyleSheet(f"color: {TEXT_COLOR};")
            self.feedback_label.setText("Press Escape to close")
            self.feedback_label.setStyleSheet(f"color: {TEXT_COLOR};")
            self.progress_label.setText("")

        # Save CSV
        self.save_results()

    def save_results(self):
        """Save trial results to CSV."""
        if not self.results:
            return

        os.makedirs(os.path.dirname(self.output_file) or ".", exist_ok=True)

        with open(self.output_file, "w", newline="") as f:
            # Metadata header
            f.write(f"# stroop_test_results\n")
            f.write(f"# participant_id: {self.params.participant_id}\n")
            f.write(f"# condition: {self.params.condition}\n")
            f.write(f"# timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            f.write(f"# n_trials: {self.params.n_trials}\n")
            f.write(f"# congruent_pct: {self.params.congruent_pct}\n")
            f.write(f"# stimulus_time_ms: {self.params.stimulus_time_ms}\n")
            f.write(f"# iti_ms: {self.params.iti_ms}\n")
            f.write("#\n")

            writer = csv.writer(f)
            writer.writerow([
                "participant_id", "condition", "trial", "word", "ink_color",
                "congruent", "response_key", "correct", "response_time_ms",
                "timestamp_sec", "stimulus_datetime",
            ])

            for t in self.results:
                writer.writerow([
                    self.params.participant_id, self.params.condition,
                    t.number, t.word, t.ink_color, t.congruent,
                    t.response_key, t.correct,
                    t.response_time_ms, t.timestamp_sec, t.stimulus_datetime,
                ])

        print(f"\nStroop results saved to: {self.output_file}")


# ======================================================================
# Entry Point
# ======================================================================

def run_stroop(n_trials=None, output_file=None, fullscreen=True):
    """
    Launch the Stroop test GUI.

    Parameters
    ----------
    n_trials : int, optional
        Override default trial count
    output_file : str, optional
        Output CSV path (auto-generated if None)
    fullscreen : bool
        Start in fullscreen mode
    """
    app = QApplication.instance() or QApplication(sys.argv)

    # Default params
    params = StroopParams()
    if n_trials is not None:
        params.n_trials = n_trials

    # Show settings dialog
    dialog = StroopSettingsDialog(params)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        print("Stroop test cancelled.")
        return None

    params = dialog.get_params()

    # Generate output filename
    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("data", exist_ok=True)
        pid = params.participant_id
        output_file = f"data/{pid}_stroop_results_{ts}.csv"

    # Create and show test window
    window = StroopTestWindow(params, output_file)
    if fullscreen:
        window.showFullScreen()
    else:
        window.resize(1024, 768)
        window.show()

    # Start the test after the window is visible
    QTimer.singleShot(500, window.start_test)

    app.exec()
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stroop Color-Word Test (PyQt6 GUI)",
        epilog="Task: Press the key matching the INK COLOR, not the word."
    )
    parser.add_argument("--trials", type=int, default=None,
                        help="Number of trials (default: set in GUI)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV path (default: auto-generate)")
    parser.add_argument("--windowed", action="store_true",
                        help="Run in windowed mode instead of fullscreen")
    args = parser.parse_args()

    run_stroop(
        n_trials=args.trials,
        output_file=args.output,
        fullscreen=not args.windowed,
    )
