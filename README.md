# Warp AV — Autonomous Vehicle Software Platform

Autonomous cargo vehicle software stack running in CARLA simulation with ROS 2.

## Quick Start

### Prerequisites
- Ubuntu 22.04
- NVIDIA GPU with drivers installed
- Python 3.10+
- ROS 2 Humble

### 1. Install ROS 2 Humble
```bash
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list'
sudo apt update
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions python3-pip
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2. Install CARLA 0.9.15
```bash
# Download CARLA
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1AF1527DE64CB8D9
sudo add-apt-repository "deb [arch=amd64] http://dist.carla.org/carla focal main"
sudo apt update
sudo apt install -y carla-simulator

# Or download tar.gz from https://github.com/carla-simulator/carla/releases/tag/0.9.15
# Extract to ~/carla
```

### 3. Install Python Dependencies
```bash
cd warp-av
pip install -r requirements.txt
```

### 4. Build ROS 2 Workspace
```bash
cd warp-av
colcon build
source install/setup.bash
```

### 5. Launch Everything
```bash
# Terminal 1: Start CARLA
cd ~/carla && ./CarlaUE4.sh

# Terminal 2: Launch autonomy stack
ros2 launch warp_av full_stack.launch.py

# Terminal 3: Open operator console
cd src/console && python3 -m http.server 8080
# Open http://localhost:8080 in browser
```

### 6. Run a Mission
Open the operator console and click "Start Mission", or:
```bash
ros2 topic pub /mission/goal std_msgs/String '{"data": "destination_1"}' --once
```

## Architecture
See [architecture/README.md](architecture/README.md)

## Scenarios
See [scenarios/README.md](scenarios/README.md)
