#!/bin/bash
# ============================================================
# Warp AV Setup Script — Ubuntu 22.04 + NVIDIA GPU
# Run this ONCE on a fresh machine
# ============================================================

set -e  # Stop on any error

echo "============================================"
echo "  Warp AV Setup"
echo "============================================"

# --- Step 1: Check NVIDIA ---
echo ""
echo "[1/6] Checking NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first:"
    echo "  sudo apt install nvidia-driver-535"
    echo "  Then reboot and run this script again."
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "✓ NVIDIA GPU found"

# --- Step 2: System packages ---
echo ""
echo "[2/6] Installing system packages..."
sudo apt update
sudo apt install -y \
    python3-pip python3-venv \
    libpng-dev libjpeg-dev libtiff-dev \
    wget unzip git

# --- Step 3: Python virtual environment ---
echo ""
echo "[3/6] Creating Python environment..."
cd "$(dirname "$0")"  # Go to script directory (warp-av/)
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install \
    numpy \
    opencv-python \
    flask \
    flask-socketio \
    gevent \
    pyyaml \
    jsonlines \
    shapely \
    transforms3d \
    requests

echo "✓ Python packages installed"

# --- Step 4: Download CARLA ---
echo ""
echo "[4/6] Downloading CARLA 0.9.15..."
echo "This is ~15 GB. It will take a while."

CARLA_DIR="$HOME/carla"

if [ -d "$CARLA_DIR" ] && [ -f "$CARLA_DIR/CarlaUE4.sh" ]; then
    echo "✓ CARLA already exists at $CARLA_DIR"
else
    mkdir -p "$CARLA_DIR"
    cd /tmp

    # Download CARLA 0.9.15
    if [ ! -f "CARLA_0.9.15.tar.gz" ]; then
        echo "Downloading CARLA..."
        wget -q --show-progress \
            https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
    fi

    echo "Extracting CARLA to $CARLA_DIR..."
    tar -xzf CARLA_0.9.15.tar.gz -C "$CARLA_DIR"
    echo "✓ CARLA extracted"
fi

# --- Step 5: Install CARLA Python API ---
echo ""
echo "[5/6] Installing CARLA Python API..."
cd "$(dirname "$0")"
source venv/bin/activate

# Find and install the CARLA .whl or .egg
CARLA_EGG=$(find "$CARLA_DIR" -name "carla-*-py3*" -type f 2>/dev/null | head -1)
CARLA_WHL=$(find "$CARLA_DIR" -name "carla-*.whl" -type f 2>/dev/null | head -1)

if [ -n "$CARLA_WHL" ]; then
    pip install "$CARLA_WHL"
    echo "✓ CARLA Python API installed from wheel"
elif [ -n "$CARLA_EGG" ]; then
    # Add egg to path instead
    echo "export PYTHONPATH=\$PYTHONPATH:$CARLA_EGG" >> venv/bin/activate
    echo "✓ CARLA Python egg added to path: $CARLA_EGG"
else
    echo "WARNING: Could not find CARLA Python package."
    echo "You may need to install it manually:"
    echo "  pip install carla==0.9.15"
    pip install carla==0.9.15 || echo "pip install failed — try manual install"
fi

# Also need CARLA's agents module for route planning
CARLA_AGENTS=$(find "$CARLA_DIR" -path "*/PythonAPI/carla" -type d 2>/dev/null | head -1)
if [ -n "$CARLA_AGENTS" ]; then
    echo "export PYTHONPATH=\$PYTHONPATH:$CARLA_AGENTS" >> venv/bin/activate
    # Also add the parent so "from agents.navigation..." works
    AGENTS_PARENT=$(dirname "$CARLA_AGENTS")
    echo "export PYTHONPATH=\$PYTHONPATH:$AGENTS_PARENT" >> venv/bin/activate
    echo "✓ CARLA agents module added to path"
else
    echo "WARNING: CARLA agents directory not found."
    echo "Route planning may not work. Check $CARLA_DIR/PythonAPI/"
fi

# --- Step 6: Create launch scripts ---
echo ""
echo "[6/6] Creating launch scripts..."
cd "$(dirname "$0")"

cat > start_carla.sh << 'SCRIPT'
#!/bin/bash
echo "Starting CARLA simulator..."
echo "Wait for the window to appear (takes 30-60 seconds)"
echo "Press Ctrl+C to stop"
cd ~/carla
./CarlaUE4.sh -quality-level=Low -prefernvidia
SCRIPT
chmod +x start_carla.sh

cat > start_av.sh << 'SCRIPT'
#!/bin/bash
echo "Starting Warp AV autonomy system..."
cd "$(dirname "$0")"
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 -m warp_av.main
SCRIPT
chmod +x start_av.sh

cat > start_console.sh << 'SCRIPT'
#!/bin/bash
echo "Opening operator console..."
echo "Go to http://localhost:8080 in your browser"
cd "$(dirname "$0")/src/warp_av/console"
python3 -m http.server 8080
SCRIPT
chmod +x start_console.sh

# --- Done ---
echo ""
echo "============================================"
echo "  ✓ SETUP COMPLETE"
echo "============================================"
echo ""
echo "To run the system, open 3 terminals:"
echo ""
echo "  Terminal 1:  ./start_carla.sh"
echo "  (wait for CARLA window to appear)"
echo ""
echo "  Terminal 2:  ./start_av.sh"
echo "  (starts the autonomy system)"
echo ""
echo "  Terminal 3:  ./start_console.sh"
echo "  (open http://localhost:8080 in browser)"
echo ""
echo "Then use the console to start a mission!"
echo ""
