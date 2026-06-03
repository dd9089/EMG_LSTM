# EMG  Gesture Classification & Arm Teleoperation

Real-time system that uses a Myo armband to simultaneously control a Robotiq 2F-85 gripper (via EMG gesture classification) .

---

## How It Works

```
Myo EMG (200 Hz) → BiLSTM classifier → Robotiq 2F-85 gripper position + effort
```

**EMG pipeline:**
- Sliding window of EMG samples fed into a trained bidirectional LSTM
- Probability averaging over the last 10 windows for smoother output
- Majority voting over the last 10 predictions for stability
- Gripper command sent only when gesture changes and confidence > 60%

---

## Requirements

### Hardware
- Myo armband
- Kinova robot arm (default IP: `192.168.1.10`)
- Robotiq 2F-85 gripper

### Software
- Ubuntu 22.04
- ROS2 Humble
- Python 3.11+

# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate biorobotics


### ROS2 workspace
```bash
git clone git@git.rc.rit.edu:kxs8997/robotiq_gripper_ur5e.git ~/ur5e/robotiq_gripper_ur5e
cd ~/ur5e/robotiq_gripper_ur5e
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

### ROS2 packages
```bash
sudo apt install ros-humble-moveit ros-humble-ur-robot-driver ros-humble-control-msgs
```

---

## Setup UR5e for operation

### Turning on the UR5e
click and hold the powert button in the teach pendant

when screen loads click on the red dot saying 'powerd off' on the bottom left. 

Then click on the button 'Turn on', then the same button now labled 'activate'

Finally click exit

### Choosing program for gripper connection
Click on the button labled 'load program'

Select the file named 'grasp'

hit ok

### If not using Configured lab laptop
You will need to change the IP address if you are not using the laptop that is configured already

go to the third tab and click on the ur caps all the way at the bottom

there you will see a host ip and host name both need to be changed to your pc's IP address

### Staring the program to connect gripper
once you are done with Terminal 1 in the startup sequence you need to hit the play button 

## Startup Sequence

You need **4 terminals** total. Start them in order and keep each running.

### Terminal 1 — Robot driver + MoveIt + RViz
```bash
cd ~/EMG_LSTM/src
Ros2 launch kinova_gen3_7dof_robotiq_2f_85_moveit_config robot.launch.py robot_ip:=192.168.1.10 use_internal_bus_gripper_comm:=true
```
Wait ~15 seconds for RViz to open with the robot model.


### Terminal 2 — Myo streaming
```bash
python myo_interface.py --select
```
Wait for `Myo streamer started!`. Verify both LSL streams are up:

Expected output:
```
Myo_EMG: 8 ch @ 200.0 Hz
```

### Terminal 3 — EMG classifier + IMU teleoperation
```bash
source /opt/ros/humble/setup.bash
python3 LSTM_RealTime_Classifier.py
```

---

## Stopping Everything

```bash
Ctr+C on each terminal
```

---

## Usage

```
Gripper (EMG gestures):
  Relax arm          → baseline   open        0 N
  Light squeeze      → soft       pos 0.30   20 N
  Medium squeeze     → medium     pos 0.56  100 N
  Hard/full squeeze  → hard       pos 0.79  200 N

Controls:
  Ctrl+C             → stop and print session summary
```

---

## CLI Arguments

| Argument      | Default                                    | Description                                      |
|---------------|--------------------------------------------|--------------------------------------------------|
| `--model`     | `saved_models/best_emg_lstm_model.pth`     | Path to trained LSTM model                       |
| `--scaler`    | `saved_models/emg_scaler.pkl`              | Path to fitted scaler                            |
| `--emg`       | `Myo_EMG`                                  | EMG LSL stream name                              |

---

### Gripper effort
Effort values map to the Robotiq 2F-85 force range (20–235 N):

| Gesture  | Position | Effort | Use case                     |
|----------|----------|--------|------------------------------|
| baseline | 0.0      | 0 N   | open / idle                  |
| soft     | 0.30     | 20 N   | fragile or compliant objects |
| medium   | 0.56     | 100 N  | general rigid objects        |
| hard     | 0.79     | 200 N  | firm hold, won't release     |

Test effort levels without running the classifier:
```bash
ros2 action send_goal /robotiq_gripper_controller/gripper_cmd \
  control_msgs/action/GripperCommand \
  "{command: {position: 0.56, max_effort: 100.0}}"
```

### Confidence threshold
The classifier only sends a gripper command when confidence > 60%. To change, edit `maybe_send_gripper()`:
```python
if confidence > 0.6:  # adjust this threshold
```

---

## Model

The trained model checkpoint (`best_emg_lstm_model.pth`) must contain:

| Key                | Description                                                        |
|--------------------|--------------------------------------------------------------------|
| `model_config`     | dict: `input_size`, `hidden_size`, `num_layers`, `num_classes`, `dropout`, `bidirectional` |
| `model_state_dict` | trained weights                                                    |
| `test_accuracy`    | float                                                              |
| `window_size`      | number of EMG samples per inference window                         |
| `gestures`         | list of class names e.g. `['baseline', 'soft', 'medium', 'hard']` |

Place model and scaler at:
```
saved_models/
├── best_emg_lstm_model.pth
└── emg_scaler.pkl
```

---

## Architecture

```
myo_interface.py
├── LSL stream: Myo_EMG (200 Hz, 8 ch)
└── LSL stream: Myo_IMU  (50 Hz, 10 ch)

LSTM_RealTime_Classifier.py
├── LSLEMGClassifier          main loop, coordinates everything
    ├── EMG_LSTM              BiLSTM model definition
    ├── predict()             windowed inference + smoothing
    └── maybe_send_gripper()  debounced gripper dispatch (subprocess)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| LSL stream not found | Make sure `myo_interface.py` is running and armband is connected |
| Gripper not responding | Terminal 3 (gripper launch) must be running before any gripper commands |
| Gripper "No such file or directory" | Terminal 1 must be running first — it creates `/tmp/ttyUR` |
| Gripper "Requested 8 bytes, got 0" | Check Teach Pendant: Tool Communication enabled, 115200 baud, 24V |
---

## System Status Checks

```bash
# Active ROS nodes
ros2 node list

# Controller states
ros2 control list_controllers

# Key topics
ros2 topic list | grep -E "(planning_scene|joint_states|gripper|servo)"

# Gripper joint state
ros2 topic echo /joint_states --once | grep -A2 robotiq
```

---

## Authors

Diego Diaz, David Reich — RIT BioRobotics / EEEE536 Teleoperation Project

