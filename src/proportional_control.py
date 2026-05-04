"""
BioRobotics Lab 1 - EMG Proportional Control Demo
==================================================

This demo shows how EMG signals can be used for proportional control,
which is a fundamental concept in myoelectric prosthetics.

The idea is simple: muscle activation → EMG amplitude → control signal

In this demo:
- Contract your forearm muscles to move a bar/cursor on screen
- Different channels can control different axes
- This is the basis of myoelectric prosthetic control!

Usage:
    python proportional_control.py              # Auto-detect stream
    python proportional_control.py --mock       # Use simulated EMG
    python proportional_control.py --stream Myo # Connect to specific stream

Author: BioRobotics Course
Updated: 2025
"""

import sys
import time
import threading
from collections import deque
from dataclasses import dataclass
import numpy as np

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSlider, QComboBox, QGroupBox, QGridLayout,
        QProgressBar
    )
    from PyQt6.QtCore import QTimer, Qt
    from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QBrush
    import pyqtgraph as pg
    HAS_GUI = True
except ImportError:
    HAS_GUI = False

try:
    import pylsl
    HAS_LSL = True
except ImportError:
    HAS_LSL = False


# pylsl API compatibility helpers
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
class ControlConfig:
    """Configuration for proportional control."""
    gain: float = 1.0           # Amplification of EMG signal
    smoothing: float = 0.1      # Smoothing factor (0-1, higher = smoother)
    threshold: float = 0.1      # Minimum activation threshold
    max_value: float = 1.0      # Maximum control output


class EMGProcessor:
    """Real-time EMG processing for control."""
    
    def __init__(self, n_channels: int = 8, window_size: int = 50):
        self.n_channels = n_channels
        self.window_size = window_size
        self.buffers = [deque(maxlen=window_size) for _ in range(n_channels)]
        self.smoothed = np.zeros(n_channels)
        self.baseline = np.zeros(n_channels)
        self.calibrated = False
    
    def add_sample(self, sample: list):
        """Add a new sample and compute control values."""
        for i, val in enumerate(sample[:self.n_channels]):
            self.buffers[i].append(val)
    
    def get_activation(self, channel: int = 0, alpha: float = 0.1) -> float:
        """
        Get the activation level for a channel.
        
        Uses RMS and exponential smoothing.
        """
        if len(self.buffers[channel]) < 10:
            return 0.0
        
        # Compute RMS
        data = np.array(self.buffers[channel])
        if self.calibrated:
            data = data - self.baseline[channel]
        rms = np.sqrt(np.mean(data ** 2))
        
        # Exponential smoothing
        self.smoothed[channel] = alpha * rms + (1 - alpha) * self.smoothed[channel]
        
        return self.smoothed[channel]
    
    def calibrate(self, duration: float = 2.0):
        """Calibrate baseline (call while at rest)."""
        for i in range(self.n_channels):
            if len(self.buffers[i]) > 10:
                self.baseline[i] = np.mean(list(self.buffers[i]))
        self.calibrated = True
        print("Calibration complete")


if HAS_GUI and HAS_LSL:
    
    class ControlVisualization(QWidget):
        """Widget for visualizing control output."""
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self.control_value = 0.0  # -1 to 1 for bidirectional, 0 to 1 for unidirectional
            self.target_value = 0.5   # Target to reach
            self.mode = 'bar'  # 'bar', 'cursor', or 'target'
            self.setMinimumSize(400, 200)
        
        def set_value(self, value: float):
            """Set the control value (0-1)."""
            self.control_value = np.clip(value, 0, 1)
            self.update()
        
        def set_target(self, value: float):
            """Set target value for target tracking mode."""
            self.target_value = np.clip(value, 0, 1)
            self.update()
        
        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            w, h = self.width(), self.height()
            margin = 20
            
            if self.mode == 'bar':
                self._draw_bar(painter, w, h, margin)
            elif self.mode == 'cursor':
                self._draw_cursor(painter, w, h, margin)
            elif self.mode == 'target':
                self._draw_target(painter, w, h, margin)
        
        def _draw_bar(self, painter, w, h, margin):
            """Draw a vertical bar indicator."""
            bar_width = 80
            bar_x = w // 2 - bar_width // 2
            bar_height = h - 2 * margin
            
            # Background
            painter.setPen(QPen(QColor(100, 100, 100), 2))
            painter.setBrush(QBrush(QColor(50, 50, 50)))
            painter.drawRect(bar_x, margin, bar_width, bar_height)
            
            # Fill based on control value
            fill_height = int(bar_height * self.control_value)
            color = QColor(0, int(255 * self.control_value), int(255 * (1 - self.control_value)))
            painter.setBrush(QBrush(color))
            painter.drawRect(bar_x, margin + bar_height - fill_height, bar_width, fill_height)
            
            # Value text
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.setFont(QFont('Arial', 16, QFont.Weight.Bold))
            painter.drawText(bar_x, h - 5, f'{self.control_value:.2f}')
        
        def _draw_cursor(self, painter, w, h, margin):
            """Draw a horizontal cursor."""
            track_y = h // 2
            track_width = w - 2 * margin
            
            # Track
            painter.setPen(QPen(QColor(100, 100, 100), 4))
            painter.drawLine(margin, track_y, w - margin, track_y)
            
            # Cursor
            cursor_x = margin + int(track_width * self.control_value)
            painter.setPen(QPen(QColor(0, 200, 0), 2))
            painter.setBrush(QBrush(QColor(0, 255, 0)))
            painter.drawEllipse(cursor_x - 15, track_y - 15, 30, 30)
        
        def _draw_target(self, painter, w, h, margin):
            """Draw cursor with target to track."""
            track_y = h // 2
            track_width = w - 2 * margin
            
            # Track
            painter.setPen(QPen(QColor(100, 100, 100), 4))
            painter.drawLine(margin, track_y, w - margin, track_y)
            
            # Target zone
            target_x = margin + int(track_width * self.target_value)
            painter.setPen(QPen(QColor(255, 200, 0), 2))
            painter.setBrush(QBrush(QColor(255, 200, 0, 100)))
            painter.drawRect(target_x - 20, track_y - 30, 40, 60)
            
            # Cursor
            cursor_x = margin + int(track_width * self.control_value)
            error = abs(self.control_value - self.target_value)
            if error < 0.05:
                color = QColor(0, 255, 0)  # Green when on target
            elif error < 0.15:
                color = QColor(255, 255, 0)  # Yellow when close
            else:
                color = QColor(255, 100, 100)  # Red when far
            
            painter.setPen(QPen(color, 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(cursor_x - 15, track_y - 15, 30, 30)
            
            # Score text
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.setFont(QFont('Arial', 12))
            painter.drawText(margin, h - 10, f'Error: {error:.3f}')


    class ProportionalControlDemo(QMainWindow):
        """Main window for the proportional control demo."""
        
        def __init__(self):
            super().__init__()
            self.setWindowTitle("EMG Proportional Control Demo")
            self.setGeometry(100, 100, 800, 600)
            
            # State
            self.inlet = None
            self.processor = None
            self.running = False
            self.selected_channel = 0
            self.config = ControlConfig()
            
            self.setup_ui()
            
            # Update timer
            self.timer = QTimer()
            self.timer.timeout.connect(self.update)
        
        def setup_ui(self):
            """Setup the user interface."""
            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)
            
            # === Control Panel ===
            control_group = QGroupBox("Settings")
            control_layout = QGridLayout(control_group)
            
            # Stream selection
            control_layout.addWidget(QLabel("Stream:"), 0, 0)
            self.stream_combo = QComboBox()
            control_layout.addWidget(self.stream_combo, 0, 1)
            
            self.refresh_btn = QPushButton("Refresh")
            self.refresh_btn.clicked.connect(self.refresh_streams)
            control_layout.addWidget(self.refresh_btn, 0, 2)
            
            self.connect_btn = QPushButton("Connect")
            self.connect_btn.clicked.connect(self.toggle_connection)
            control_layout.addWidget(self.connect_btn, 0, 3)
            
            # Channel selection
            control_layout.addWidget(QLabel("Channel:"), 1, 0)
            self.channel_combo = QComboBox()
            for i in range(8):
                self.channel_combo.addItem(f"Channel {i+1}", i)
            self.channel_combo.currentIndexChanged.connect(
                lambda idx: setattr(self, 'selected_channel', idx)
            )
            control_layout.addWidget(self.channel_combo, 1, 1)
            
            # Calibrate button
            self.calibrate_btn = QPushButton("Calibrate (Rest)")
            self.calibrate_btn.clicked.connect(self.calibrate)
            self.calibrate_btn.setEnabled(False)
            control_layout.addWidget(self.calibrate_btn, 1, 2)
            
            # Mode selection
            control_layout.addWidget(QLabel("Mode:"), 1, 3)
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["Bar", "Cursor", "Target Tracking"])
            self.mode_combo.currentTextChanged.connect(self.change_mode)
            control_layout.addWidget(self.mode_combo, 1, 4)
            
            # Gain slider
            control_layout.addWidget(QLabel("Gain:"), 2, 0)
            self.gain_slider = QSlider(Qt.Orientation.Horizontal)
            self.gain_slider.setRange(1, 100)
            self.gain_slider.setValue(10)
            self.gain_slider.valueChanged.connect(
                lambda v: setattr(self.config, 'gain', v / 10.0)
            )
            control_layout.addWidget(self.gain_slider, 2, 1, 1, 2)
            self.gain_label = QLabel("1.0")
            control_layout.addWidget(self.gain_label, 2, 3)
            
            layout.addWidget(control_group)
            
            # === Visualization ===
            self.viz = ControlVisualization()
            layout.addWidget(self.viz, stretch=1)
            
            # === Channel bars ===
            bars_group = QGroupBox("All Channels")
            bars_layout = QHBoxLayout(bars_group)
            self.channel_bars = []
            for i in range(8):
                vbox = QVBoxLayout()
                bar = QProgressBar()
                bar.setOrientation(Qt.Orientation.Vertical)
                bar.setRange(0, 100)
                bar.setValue(0)
                bar.setMinimumHeight(100)
                vbox.addWidget(bar)
                vbox.addWidget(QLabel(f"Ch{i+1}"))
                bars_layout.addLayout(vbox)
                self.channel_bars.append(bar)
            layout.addWidget(bars_group)
            
            # === Instructions ===
            instructions = QLabel(
                "<b>Instructions:</b><br>"
                "1. Connect to your EMG stream (or use Mock for testing)<br>"
                "2. Rest your arm and click 'Calibrate'<br>"
                "3. Contract your forearm muscles to control the bar<br>"
                "4. Try different modes and adjust gain as needed"
            )
            instructions.setWordWrap(True)
            layout.addWidget(instructions)
            
            self.refresh_streams()
        
        def refresh_streams(self):
            """Scan for LSL streams."""
            self.stream_combo.clear()
            
            # Add mock option
            self.stream_combo.addItem("Mock EMG (Testing)", "mock")
            
            # Scan for real streams
            streams = _resolve_streams(1.0)
            for stream in streams:
                name = stream.name()
                self.stream_combo.addItem(f"{name} ({stream.type()})", name)
        
        def toggle_connection(self):
            """Connect or disconnect."""
            if self.running:
                self.disconnect()
            else:
                self.connect()
        
        def connect(self):
            """Connect to the selected stream."""
            stream_name = self.stream_combo.currentData()
            
            if stream_name == "mock":
                # Start mock data generator
                self.start_mock()
            else:
                # Connect to real stream
                streams = _resolve_byprop("name", stream_name, timeout=2.0)
                if not streams:
                    return
                
                self.inlet = pylsl.StreamInlet(streams[0], max_buflen=360)
                n_channels = streams[0].channel_count()
                self.processor = EMGProcessor(n_channels)
            
            self.running = True
            self.timer.start(16)  # ~60 Hz
            self.connect_btn.setText("Disconnect")
            self.calibrate_btn.setEnabled(True)
        
        def start_mock(self):
            """Start mock data generation."""
            self.processor = EMGProcessor(8)
            self.mock_t = 0
            self.inlet = None
        
        def disconnect(self):
            """Disconnect from stream."""
            self.running = False
            self.timer.stop()
            self.inlet = None
            self.connect_btn.setText("Connect")
            self.calibrate_btn.setEnabled(False)
        
        def calibrate(self):
            """Calibrate baseline."""
            if self.processor:
                self.processor.calibrate()
        
        def change_mode(self, mode_text):
            """Change visualization mode."""
            mode_map = {"Bar": "bar", "Cursor": "cursor", "Target Tracking": "target"}
            self.viz.mode = mode_map.get(mode_text, "bar")
            self.viz.update()
        
        def update(self):
            """Update loop."""
            if not self.running or not self.processor:
                return
            
            # Get data
            if self.inlet:
                samples, _ = self.inlet.pull_chunk(timeout=0.0)
                for sample in samples:
                    self.processor.add_sample(sample)
            else:
                # Generate mock data
                self.mock_t += 0.016
                sample = []
                for ch in range(8):
                    # Simulate some activation on selected channel
                    base = np.random.randn() * 5
                    if ch == self.selected_channel:
                        # Add time-varying activation
                        activation = 30 * (0.5 + 0.5 * np.sin(self.mock_t * 2))
                        base += activation
                    sample.append(base)
                self.processor.add_sample(sample)
            
            # Update gain label
            self.gain_label.setText(f"{self.config.gain:.1f}")
            
            # Update channel bars
            for i in range(min(8, len(self.channel_bars))):
                activation = self.processor.get_activation(i) * self.config.gain
                self.channel_bars[i].setValue(int(min(100, activation)))
            
            # Update main visualization
            activation = self.processor.get_activation(self.selected_channel)
            control = np.clip(activation * self.config.gain / 100, 0, 1)
            self.viz.set_value(control)
            
            # Update target in tracking mode
            if self.viz.mode == 'target':
                # Slowly moving target
                t = time.time()
                target = 0.5 + 0.4 * np.sin(t * 0.5)
                self.viz.set_target(target)
        
        def closeEvent(self, event):
            self.disconnect()
            event.accept()


def main():
    """Main entry point."""
    if not HAS_GUI:
        print("Error: PyQt6 not available. Install with: pip install PyQt6 pyqtgraph")
        return 1
    
    if not HAS_LSL:
        print("Error: pylsl not available. Install with: pip install pylsl")
        return 1
    
    import argparse
    parser = argparse.ArgumentParser(description="EMG Proportional Control Demo")
    parser.add_argument("--stream", help="Stream name to connect to")
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    window = ProportionalControlDemo()
    window.show()
    
    if args.mock:
        window.stream_combo.setCurrentIndex(0)  # Select mock
        window.connect()
    elif args.stream:
        for i in range(window.stream_combo.count()):
            if args.stream in window.stream_combo.itemText(i):
                window.stream_combo.setCurrentIndex(i)
                window.connect()
                break
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
