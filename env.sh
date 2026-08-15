#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$PROJECT_DIR/venv/bin/activate"

export PYTHONPATH="$PYTHONPATH:$HOME/carla/PythonAPI/carla"
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR/src"

echo "Warp AV environment ready"
echo "Project: $PROJECT_DIR"
