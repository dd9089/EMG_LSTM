#!/usr/bin/env python3
"""
EMG Gesture Classification with LSL Stream
===========================================

Receives EMG data from the lab's Myo LSL stream and classifies gestures
in real-time using your trained LSTM model.

Usage:
    # Terminal 1: Start Myo streaming
    python myo_interface.py --select
    
    # Terminal 2: Start gesture classification
    python lsl_lstm_classifier.py

Author: Diego Diaz
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import pickle
import subprocess
import threading
from collections import deque, Counter
import time
import sys

# Import LSL
try:
    import pylsl
    HAS_LSL = True
except ImportError:
    print("pylsl not installed!")
    print("Install with: pip install pylsl")
    sys.exit(1)


# =============================================================================
# MODEL DEFINITION
# =============================================================================

class EMG_LSTM(nn.Module):
    """LSTM network for EMG gesture classification."""
    
    def __init__(self, input_size=8, hidden_size=64, num_layers=3, 
                 num_classes=4, dropout=0.75, bidirectional=True):
        super(EMG_LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, num_classes)
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        if self.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]
        
        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)
        
        return out


# =============================================================================
# LSL EMG CLASSIFIER
# =============================================================================

class LSLEMGClassifier:
    """
    Real-time gesture classifier that receives EMG from LSL stream.
    """
    
    def __init__(self, model_path='saved_models/best_emg_lstm_model.pth',
                 scaler_path='saved_models/emg_scaler.pkl',
                 stream_name='Myo_EMG'):
        
        print("\n" + "="*70)
        print("LSL EMG GESTURE CLASSIFIER")
        print("="*70)
        
        # Load model
        print("\nLoading trained model...")
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['model_config']
        
        self.model = EMG_LSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout=config['dropout'],
            bidirectional=config['bidirectional']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded")
        print(f"  Architecture: {config['num_layers']}-layer BiLSTM")
        print(f"  Test accuracy: {checkpoint['test_accuracy']*100:.2f}%")
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"Scaler loaded")
        
        # Configuration
        self.window_size = checkpoint['window_size']
        self.gestures = checkpoint['gestures']
        self.stream_name = stream_name
        
        # EMG buffer (sliding window)
        self.emg_buffer = deque(maxlen=self.window_size)
        
        # Prediction history (temporal smoothing via voting)
        self.prediction_history = deque(maxlen=10)
        
        # Probability averaging (smoother confidence)
        self.probability_history = deque(maxlen=10)
        
        # LSL inlet
        self.inlet = None
        
        # Gripper command state (debounce)
        self.last_sent_gesture = None
        self.last_sent_time = 0
        self._gripper_lock = threading.Lock()

        # Statistics
        self.total_predictions = 0
        self.gesture_counts = {g: 0 for g in self.gestures}
        self.start_time = time.time()
        
        print(f"\nConfiguration:")
        print(f"  Window size: {self.window_size} samples")
        print(f"  Gestures: {self.gestures}")
        print(f"  LSL stream: {stream_name}")
        print("="*70)
    
    def connect_to_stream(self, timeout=10.0):
        """Connect to LSL EMG stream."""
        
        print(f"\nSearching for LSL stream '{self.stream_name}'...")
        print("(Make sure myo_interface.py is running!)")
        
        streams = pylsl.resolve_byprop('name', self.stream_name, timeout=timeout)
        
        if not streams:
            print(f"\nNo stream named '{self.stream_name}' found!")
            print("\nMake sure to:")
            print("  1. Run: python myo_interface.py --select")
            print("  2. Connect to your Myo armband")
            print("  3. Wait for 'Myo streamer started!' message")
            print("  4. Then run this script")
            return False
        
        self.inlet = pylsl.StreamInlet(streams[0], max_buflen=3600)
        
        info = self.inlet.info()
        print(f"\nConnected to stream!")
        print(f"  Name: {info.name()}")
        print(f"  Type: {info.type()}")
        print(f"  Channels: {info.channel_count()}")
        print(f"  Sample rate: {info.nominal_srate()} Hz")
        
        return True
    
    def predict(self):
        """Predict gesture from current buffer with probability averaging and majority voting."""
        
        # Convert buffer to array and normalize
        window = np.array(self.emg_buffer)
        window_normalized = self.scaler.transform(window)
        
        # Convert to tensor and run model
        window_tensor = torch.FloatTensor(window_normalized).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(window_tensor)
            probabilities = torch.softmax(output, dim=1).numpy()[0]
        
        # Accumulate probability history and average
        self.probability_history.append(probabilities)
        avg_probabilities = np.mean(list(self.probability_history), axis=0)
        
        # Get prediction from averaged probabilities
        pred_idx = np.argmax(avg_probabilities)
        gesture = self.gestures[pred_idx]
        confidence = avg_probabilities[pred_idx]
        
        # Majority vote over recent predictions for extra stability
        self.prediction_history.append(gesture)
        if len(self.prediction_history) >= 3:
            gesture_counts = Counter(self.prediction_history)
            smoothed_gesture = gesture_counts.most_common(1)[0][0]
        else:
            smoothed_gesture = gesture
        
        # Update statistics
        self.total_predictions += 1
        self.gesture_counts[smoothed_gesture] += 1
        
        return {
            'gesture': smoothed_gesture,
            'confidence': confidence,
            'probabilities': avg_probabilities,
            'raw_gesture': gesture,
            'raw_probabilities': probabilities
        }

    def _send_gripper_command(self, position, effort):
        cmd = [
            "ros2", "action", "send_goal",
            "/robotiq_gripper_controller/gripper_cmd",
            "control_msgs/action/GripperCommand",
            f"{{command: {{position: {position}, max_effort: {effort}}}}}"
        ]
        subprocess.run(cmd)

    def display_prediction(self, prediction):
        """Display prediction in console and dispatch gripper command if gesture changed."""
        
        gesture = prediction['gesture']
        confidence = prediction['confidence']
        probs = prediction['probabilities']

        # Gripper positions per gesture
        gripper_positions = {
            'baseline': 0.0,
            'soft':     0.3,
            'medium':   0.56,
            'hard':     0.7929,
        }

        # Gripper positions per gesture
        gripper_effort = {
            'baseline': 0.0,
            'soft':     20,
            'medium':   100,
            'hard':     200,
        }

        # Only send command when gesture changes and confidence is high enough
        # Uses a lock to avoid race conditions with the background thread
        now = time.time()
        if (gesture in gripper_positions and
                confidence > 0.6 and
                gesture != self.last_sent_gesture and
                (now - self.last_sent_time) > 0.5):

            with self._gripper_lock:
                self.last_sent_gesture = gesture
                self.last_sent_time = now

            position = gripper_positions[gesture]
            effort = gripper_effort[gesture]
            threading.Thread(
                target=self._send_gripper_command,
                args=(position, effort),
                daemon=True   # won't block Ctrl+C
            ).start()

        # ── Console display ──────────────────────────────────────────────────
        print(" " * 100, end='\r')

        bar_length = 30
        filled = int(bar_length * confidence)
        bar = '█' * filled + '░' * (bar_length - filled)

        prob_str = " | ".join([f"{g[:3]}:{p*100:4.1f}%"
                               for g, p in zip(self.gestures, probs)])

        print(f"{gesture:10s} [{bar}] {confidence*100:5.1f}% | {prob_str}",
              end='\r')
    
    def run(self):
        """Main classification loop."""
        
        if self.inlet is None:
            print("Not connected to stream! Call connect_to_stream() first.")
            return
        
        print("\n" + "="*70)
        print("REAL-TIME GESTURE CLASSIFICATION")
        print("="*70)
        print("\nPerform gestures to see predictions:")
        print("  - Relax your arm         → baseline")
        print("  - Light squeeze          → soft")
        print("  - Medium squeeze         → medium")
        print("  - Hard/full squeeze      → hard")
        print("\nPress Ctrl+C to stop")
        print("="*70 + "\n")
        
        print(f"Filling buffer (need {self.window_size} samples)...")
        
        try:
            while True:
                sample, timestamp = self.inlet.pull_sample(timeout=0.1)
                
                if sample is None:
                    print("\nNo data received. Is myo_interface.py still running?")
                    continue
                
                self.emg_buffer.append(sample[:8])
                
                if len(self.emg_buffer) == self.window_size:
                    prediction = self.predict()
                    self.display_prediction(prediction)
        
        except KeyboardInterrupt:
            print("\n\nStopping...")
        
        finally:
            self.print_summary()
    
    def print_summary(self):
        """Print session summary."""
        
        elapsed = time.time() - self.start_time
        
        print("\n\n" + "="*70)
        print("SESSION SUMMARY")
        print("="*70)
        print(f"Duration: {elapsed:.1f} seconds")
        print(f"Total predictions: {self.total_predictions}")
        if elapsed > 0:
            print(f"Prediction rate: {self.total_predictions/elapsed:.1f} Hz")
        
        if self.total_predictions > 0:
            print("\nGesture distribution:")
            for gesture in self.gestures:
                count = self.gesture_counts[gesture]
                percentage = count / self.total_predictions * 100
                bar = '█' * int(percentage / 2)
                print(f"  {gesture:10s}: {count:5d} ({percentage:5.1f}%) {bar}")
        
        print("="*70)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Real-time EMG gesture classification from LSL stream'
    )
    parser.add_argument('--model', default='saved_models/best_emg_lstm_model.pth',
                       help='Path to trained model')
    parser.add_argument('--scaler', default='saved_models/emg_scaler.pkl',
                       help='Path to scaler')
    parser.add_argument('--stream', default='Myo_EMG',
                       help='LSL stream name (default: Myo_EMG)')
    
    args = parser.parse_args()
    
    try:
        classifier = LSLEMGClassifier(
            model_path=args.model,
            scaler_path=args.scaler,
            stream_name=args.stream
        )
    except Exception as e:
        print(f"\n Error loading model: {e}")
        return 1
    
    if not classifier.connect_to_stream(timeout=10.0):
        return 1
    
    classifier.run()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())