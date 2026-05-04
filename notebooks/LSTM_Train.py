"""
EMG Gesture Classification with LSTM
=====================================

Complete implementation of LSTM-based EMG gesture classification with 
participant-level split for cross-subject generalization testing.

This script replaces the LDA/QDA section of the original lab notebook.

Author: Diego Diaz
Date: 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard libraries
import numpy as np
import pandas as pd
from pathlib import Path
import os
import glob

# Signal processing
import scipy.signal as signal
from scipy import stats
from scipy.fft import fft, fftfreq

# Machine learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

print("All imports successful")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data parameters
DATA_DIR = '../recordings'
SAMPLE_RATE = 200  # Hz
GESTURES = ['baseline', 'hard', 'medium', 'soft']

# LSTM parameters
WINDOW_SIZE = 80
OVERLAP = 0.5  
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
HIDDEN_SIZE = 64 
NUM_LAYERS = 3 
DROPOUT = 0.75
PATIENCE = 15

# Data augmentation parameters
USE_AUGMENTATION = True 
NOISE_LEVEL = 0.05 
SCALE_RANGE = 0.20 
MAX_TIME_SHIFT = 5 

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f"\nConfiguration loaded")
print(f"  Data directory: {DATA_DIR}")
print(f"  Gestures: {GESTURES}")
print(f"  Window size: {WINDOW_SIZE} samples ({WINDOW_SIZE/SAMPLE_RATE:.1f}s)")
print(f"  Data augmentation: {'ENABLED' if USE_AUGMENTATION else 'DISABLED'}")

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_emg_file(filepath):
    """
    Load a single EMG CSV file with metadata.
    
    Expected format:
    - Header lines start with '#' containing metadata
    - Data columns: timestamp, emg_1, emg_2, ..., emg_8
    
    Returns
    -------
    data : pd.DataFrame
        EMG data with columns [timestamp, emg_1, ..., emg_8]
    metadata : dict
        Metadata extracted from header
    """
    metadata = {}
    
    # Read header lines
    with open(filepath, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                break
            if ':' in line:
                key, value = line[1:].split(':', 1)
                metadata[key.strip()] = value.strip()
    
    # Read data
    data = pd.read_csv(filepath, comment='#')
    
    return data, metadata


def load_all_trials(data_dir, gestures):
    """
    Load all EMG trials from nested directory structure.
    
    Expected structure:
        data_dir/
        ├── P1/
        │   ├── baseline/
        │   │   ├── trial001_emg_timestamp.csv
        │   │   └── trial002_emg_timestamp.csv
        │   ├── hard/
        │   ├── medium/
        │   └── soft/
        ├── P2/
        └── ...
    
    Parameters
    ----------
    data_dir : str
        Path to data directory
    gestures : list
        List of gesture names
    
    Returns
    -------
    all_trials : dict
        Dictionary mapping gesture names to lists of (df, metadata) tuples
    """
    all_trials = {}
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return all_trials
    
    # Find participant directories
    participant_dirs = [d for d in os.listdir(data_dir) 
                       if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('P')]
    
    if not participant_dirs:
        print(f"No participant directories found in {data_dir}")
        return all_trials
    
    participant_dirs.sort()
    print(f"\nFound {len(participant_dirs)} participants: {', '.join(participant_dirs)}")
    
    total_files = 0
    
    # Iterate through participants and gestures
    for participant in participant_dirs:
        participant_path = os.path.join(data_dir, participant)
        
        for gesture in gestures:
            gesture_path = os.path.join(participant_path, gesture)
            
            if not os.path.exists(gesture_path):
                continue
            
            # Find all CSV files
            csv_files = glob.glob(os.path.join(gesture_path, '*.csv'))
            
            for filepath in sorted(csv_files):
                try:
                    df, meta = load_emg_file(filepath)
                    
                    # Add participant info
                    meta['participant'] = participant
                    meta['gesture'] = gesture
                    
                    # Initialize gesture list if needed
                    if gesture not in all_trials:
                        all_trials[gesture] = []
                    
                    all_trials[gesture].append((df, meta))
                    total_files += 1
                    
                except Exception as e:
                    print(f"  Error loading {participant}/{gesture}/{os.path.basename(filepath)}: {e}")
    
    print(f"\nLoaded {total_files} files total")
    
    # Summarize
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    for gesture in gestures:
        n_trials = len(all_trials.get(gesture, []))
        print(f"   {gesture}: {n_trials} trials")
    
    return all_trials


# =============================================================================
# LSTM DATA PREPARATION
# =============================================================================

def prepare_lstm_data_participant_split(all_trials, gestures, window_size=200,
                                       overlap=0.5, test_size=0.2, val_size=0.1,
                                       random_state=42):
    """
    Split EMG data by PARTICIPANT for cross-subject generalization testing.
    
    This ensures test participants are COMPLETELY UNSEEN during training.
    Tests whether the model can generalize to new users.
    
    Parameters
    ----------
    all_trials : dict
        Dictionary mapping gesture names to lists of (df, metadata) tuples
    gestures : list
        List of gesture names
    window_size : int
        Samples per window (200 = 1 second at 200 Hz)
    overlap : float
        Window overlap (0-1)
    test_size : float
        Proportion of PARTICIPANTS for test set (not trials!)
    val_size : float
        Proportion of PARTICIPANTS for validation set
    random_state : int
        Random seed for reproducibility
    
    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test), participant_info
    """
    # Collect all trials by participant
    participant_trials = {}
    
    for gesture_idx, gesture in enumerate(gestures):
        if gesture not in all_trials:
            continue
        
        for trial_data, metadata in all_trials[gesture]:
            participant = metadata.get('participant', 'unknown')
            
            if participant not in participant_trials:
                participant_trials[participant] = []
            
            participant_trials[participant].append({
                'data': trial_data,
                'gesture': gesture_idx,
                'gesture_name': gesture,
                'participant': participant
            })
    
    participants = sorted(list(participant_trials.keys()))
    
    print("\n" + "="*70)
    print("PARTICIPANT-LEVEL SPLIT (Cross-Subject Generalization)")
    print("="*70)
    print(f"\nTotal participants: {len(participants)}")
    print(f"Participants: {participants}")
    
    # Show trials per participant
    print(f"\nTrials per participant:")
    for p in participants:
        n_trials = len(participant_trials[p])
        gestures_present = set(t['gesture_name'] for t in participant_trials[p])
        print(f"  {p}: {n_trials} trials across {len(gestures_present)} gestures")
    
    # Calculate split sizes
    n_test = max(1, int(len(participants) * test_size))
    n_val = max(1, int(len(participants) * val_size))
    
    # Split participants (not trials!)
    train_val_participants, test_participants = train_test_split(
        participants,
        test_size=n_test,
        random_state=random_state
    )
    
    train_participants, val_participants = train_test_split(
        train_val_participants,
        test_size=n_val,
        random_state=random_state
    )
    
    print(f"\n Participant assignment:")
    print(f"  Train: {train_participants} ({len(train_participants)} people)")
    print(f"  Val:   {val_participants} ({len(val_participants)} people)")
    print(f"  Test:  {test_participants} ({len(test_participants)} people)")
    
    # Collect trials for each split
    train_trials = []
    val_trials = []
    test_trials = []
    
    for p in train_participants:
        train_trials.extend(participant_trials[p])
    for p in val_participants:
        val_trials.extend(participant_trials[p])
    for p in test_participants:
        test_trials.extend(participant_trials[p])
    
    print(f"\nTrials collected:")
    print(f"  Train: {len(train_trials)} trials")
    print(f"  Val:   {len(val_trials)} trials")
    print(f"  Test:  {len(test_trials)} trials")
    
    # Create windows
    def trials_to_windows(trials):
        X_list = []
        y_list = []
        step_size = int(window_size * (1 - overlap))
        
        for trial_info in trials:
            emg_data = trial_info['data'][['emg_1', 'emg_2', 'emg_3', 'emg_4',
                                           'emg_5', 'emg_6', 'emg_7', 'emg_8']].values
            
            for start_idx in range(0, len(emg_data) - window_size + 1, step_size):
                window = emg_data[start_idx:start_idx + window_size]
                X_list.append(window)
                y_list.append(trial_info['gesture'])
        
        return np.array(X_list), np.array(y_list)
    
    print(f"\nCreating windows (size={window_size}, overlap={overlap}):")
    X_train, y_train = trials_to_windows(train_trials)
    X_val, y_val = trials_to_windows(val_trials)
    X_test, y_test = trials_to_windows(test_trials)
    
    print(f"  Train: {len(X_train)} windows")
    print(f"  Val:   {len(X_val)} windows")
    print(f"  Test:  {len(X_test)} windows")
    
    participant_info = {
        'train_participants': train_participants,
        'val_participants': val_participants,
        'test_participants': test_participants,
        'all_participants': participants
    }
    
    print(f"\nCross-subject split complete!")
    print(f"  This tests TRUE generalization to unseen users")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), participant_info


# =============================================================================
# LSTM MODEL ARCHITECTURE
# =============================================================================

class EMG_LSTM(nn.Module):
    """
    LSTM network for EMG gesture classification.
    
    Architecture:
    - Input: (batch, timesteps, channels)
    - 2 LSTM layers with dropout
    - Fully connected output layer
    - Output: (batch, num_classes)
    """
    
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, 
                 num_classes=4, dropout=0.3, bidirectional=False):
        """
        Parameters
        ----------
        input_size : int
            Number of EMG channels (8 for Myo)
        hidden_size : int
            Number of LSTM hidden units
        num_layers : int
            Number of stacked LSTM layers
        num_classes : int
            Number of gesture classes
        dropout : float
            Dropout probability between LSTM layers
        bidirectional : bool
            Whether to use bidirectional LSTM (default: False)
        """
        super(EMG_LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layer
        # If bidirectional, hidden size is doubled
        fc_input_size = hidden_size * 2
        self.fc = nn.Linear(fc_input_size, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor of shape (batch, timesteps, channels)
        
        Returns
        -------
        out : torch.Tensor of shape (batch, num_classes)
        """
        # Debug: Print input shape on first call
        if not hasattr(self, '_shapes_printed'):
            print(f"\n[DEBUG] Input shape: {x.shape}")
            self._shapes_printed = True
        
        # LSTM forward pass
        # lstm_out: (batch, seq_len, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        # c_n: (num_layers * num_directions, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        if not hasattr(self, '_lstm_shapes_printed'):
            print(f"[DEBUG] lstm_out shape: {lstm_out.shape}")
            print(f"[DEBUG] h_n shape: {h_n.shape}")
            print(f"[DEBUG] Bidirectional: {self.bidirectional}")
            self._lstm_shapes_printed = True
        
        # Extract final hidden state
        if self.bidirectional:
            # For bidirectional LSTM, concatenate forward and backward hidden states
            # h_n[-2] is the forward direction of the last layer
            # h_n[-1] is the backward direction of the last layer
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
            if not hasattr(self, '_concat_printed'):
                print(f"[DEBUG] Concatenated hidden shape: {last_hidden.shape}")
                self._concat_printed = True
        else:
            # For unidirectional LSTM, just use the last layer's hidden state
            last_hidden = h_n[-1]
            if not hasattr(self, '_hidden_printed'):
                print(f"[DEBUG] Using h_n[-1] shape: {last_hidden.shape}")
                self._hidden_printed = True
        
        # Apply dropout
        last_hidden = self.dropout(last_hidden)
        
        # Fully connected layer
        out = self.fc(last_hidden)
        
        return out


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device, augment=True):
    """
    Train the model for one epoch with optional data augmentation.
    
    Parameters
    ----------
    model : nn.Module
        The LSTM model
    loader : DataLoader
        Training data loader
    criterion : nn.Module
        Loss function
    optimizer : optim.Optimizer
        Optimizer
    device : torch.device
        Device to train on
    augment : bool
        Whether to apply data augmentation (default: True)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Data augmentation for EMG signals
        if augment:
            # 1. Add Gaussian noise (simulates electrode noise and movement artifacts)
            noise = torch.randn_like(inputs) * NOISE_LEVEL
            inputs = inputs + noise
            
            # 2. Random amplitude scaling (simulates different muscle strengths and electrode coupling)
            scale = 1.0 + (torch.rand(inputs.size(0), 1, 1, device=device) * 2 - 1) * SCALE_RANGE
            inputs = inputs * scale
            
            # 3. Random time shift (simulates timing variations in gesture execution)
            # Randomly shift the window by a few samples
            if torch.rand(1).item() > 0.5:  # 50% chance to apply
                shift = torch.randint(-MAX_TIME_SHIFT, MAX_TIME_SHIFT + 1, (1,)).item()
                if shift != 0:
                    inputs = torch.roll(inputs, shifts=shift, dims=1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / total, correct / total, all_preds, all_labels


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print("EMG GESTURE CLASSIFICATION WITH LSTM")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    all_trials = load_all_trials(DATA_DIR, GESTURES)
    
    if not all_trials:
        print("\n❌ No data loaded. Please check DATA_DIR path.")
        return
    
    # Prepare LSTM data with participant split
    print("\n2. Preparing LSTM data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), participant_info = \
        prepare_lstm_data_participant_split(
            all_trials, GESTURES,
            window_size=WINDOW_SIZE,
            overlap=OVERLAP,
            test_size=0.17,   # 2 participants (~17%)
            val_size=0.25,    # 3 participants (~25%)
            random_state=RANDOM_SEED
        )
    
    print(f"\nData shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    
    # Normalize data
    print("\n3. Normalizing data...")
    X_train_flat = X_train.reshape(-1, 8)
    X_val_flat = X_val.reshape(-1, 8)
    X_test_flat = X_test.reshape(-1, 8)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"DataLoaders created (batch_size={BATCH_SIZE})")
    
    # Create model
    print("\n4. Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = EMG_LSTM(
        input_size=8,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=len(GESTURES),
        dropout=DROPOUT,
        bidirectional=True
    ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nModel parameters:")
    print(f"  input_size: 8")
    print(f"  hidden_size: {HIDDEN_SIZE}")
    print(f"  num_layers: {NUM_LAYERS}")
    print(f"  num_classes: {len(GESTURES)}")
    print(f"  dropout: {DROPOUT}")
    print(f"  bidirectional: {model.bidirectional}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Verify FC layer dimensions
    expected_fc_input = HIDDEN_SIZE * 2 if model.bidirectional else HIDDEN_SIZE
    print(f"\nFC layer verification:")
    print(f"  Bidirectional: {model.bidirectional}")
    print(f"  Expected input size: {expected_fc_input}")
    print(f"  Actual FC.in_features: {model.fc.in_features}")
    print(f"  Actual FC.out_features: {model.fc.out_features}")
    
    if model.fc.in_features != expected_fc_input:
        print(f"\nERROR: FC layer input size mismatch!")
        print(f"   Expected: {expected_fc_input}, Got: {model.fc.in_features}")
        return
    
    print(f"\nModel configured correctly for bidirectional={model.bidirectional}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\n5. Training model...")
    print("="*70)
    print(f"Data augmentation: {'ENABLED' if USE_AUGMENTATION else 'DISABLED'}")
    if USE_AUGMENTATION:
        print(f"  - Gaussian noise: {NOISE_LEVEL*100:.1f}% of signal")
        print(f"  - Amplitude scaling: ±{SCALE_RANGE*100:.1f}%")
        print(f"  - Time shift: ±{MAX_TIME_SHIFT} samples")
    print("="*70)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        # Train with augmentation
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            augment=USE_AUGMENTATION
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n Early stopping at epoch {epoch+1}")
                print(f"  Best validation: {best_val_acc:.2%} at epoch {best_epoch}")
                break
    
    print("="*70)
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nLoaded best model from epoch {best_epoch}")
    
    # Save the best model (Windows compatible paths)
    import os
    
    # Create output directory if it doesn't exist
    output_dir = 'saved_models'
    os.makedirs(output_dir, exist_ok=True)
    
    model_save_path = os.path.join(output_dir, 'best_emg_lstm_model.pth')
    scaler_save_path = os.path.join(output_dir, 'emg_scaler.pkl')
    
    print(f"\n6.5. Saving best model...")
    
    # Save model state dict
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_model_state,
        'model_config': {
            'input_size': 8,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'num_classes': len(GESTURES),
            'dropout': DROPOUT,
            'bidirectional': True
        },
        'test_accuracy': 0.0,  # Will update after evaluation
        'gestures': GESTURES,
        'window_size': WINDOW_SIZE,
        'overlap': OVERLAP,
        'sample_rate': SAMPLE_RATE,
    }, model_save_path)
    
    # Save scaler
    import pickle
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Model saved to: {model_save_path}")
    print(f"Scaler saved to: {scaler_save_path}")
    
    # Evaluate on test set
    print("\n6. Evaluating on test set...")
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    
    # Update saved model with test accuracy
    checkpoint = torch.load(model_save_path)
    checkpoint['test_accuracy'] = test_acc
    torch.save(checkpoint, model_save_path)
    print(f"\nUpdated model with test accuracy: {test_acc:.2%}")
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print('='*70)
    print(f"  Best Val Accuracy:  {best_val_acc:.2%} (epoch {best_epoch})")
    print(f"  Test Accuracy:      {test_acc:.2%}")
    print('='*70)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=GESTURES))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=GESTURES,
                yticklabels=GESTURES,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Gesture', fontsize=12)
    plt.ylabel('True Gesture', fontsize=12)
    plt.title(f'3-Layer BiLSTM Confusion Matrix - Cross-Subject (Accuracy: {test_acc:.2%})', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix_bilstm.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved to 'confusion_matrix_bilstm.png'")
    
    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, label='Train Loss', linewidth=2, alpha=0.8)
    ax1.plot(epochs, val_losses, label='Val Loss', linewidth=2, alpha=0.8)
    ax1.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, 
                label=f'Best (epoch {best_epoch})')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, label='Train Accuracy', linewidth=2, alpha=0.8)
    ax2.plot(epochs, val_accs, label='Val Accuracy', linewidth=2, alpha=0.8)
    ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, 
                label=f'Best (epoch {best_epoch})')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('training_curves_bilstm.png', dpi=300, bbox_inches='tight')
    print("Training curves saved to 'training_curves_bilstm.png'")
    
    # Cross-subject analysis
    print("\n" + "="*70)
    print("CROSS-SUBJECT GENERALIZATION ANALYSIS")
    print("="*70)
    
    print(f"\n Training Participants:   {participant_info['train_participants']}")
    print(f" Validation Participants: {participant_info['val_participants']}")
    print(f" Test Participants:       {participant_info['test_participants']}")
    
    print(f"\n Results Summary:")
    print(f"  Training on:   {len(participant_info['train_participants'])} participants")
    print(f"  Validating on: {len(participant_info['val_participants'])} participants")
    print(f"  Testing on:    {len(participant_info['test_participants'])} participants (NEW USERS)")
    
    print(f"\n Test Accuracy: {test_acc:.2%}")
    print(f"   This represents performance on COMPLETELY UNSEEN participants")
    
    # Benchmarks
    print(f"\n Cross-Subject Benchmarks (from literature):")
    print(f"  Excellent:  >80%")
    print(f"  Good:       70-80%")
    print(f"  Fair:       60-70%")
    print(f"  Challenging: <60%")
        
    print("\n" + "="*70)
    print("LSTM training and evaluation complete!")
    print("="*70)


if __name__ == "__main__":
    main()