#!/usr/bin/env bash
# ==============================================================================
# RoadVision AI — Native systemd service setup (No-Docker Alternative)
# For users preferring a direct Python virtual environment on Ubuntu ARM64
# ==============================================================================
set -euo pipefail

sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv python3-dev build-essential libgl1 libglib2.0-0

VENV_DIR="/home/ubuntu/roadvision_venv"
APP_DIR="/home/ubuntu/RoadVision-AI"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
"$VENV_DIR/bin/pip" install -r "$APP_DIR/backend/requirements.txt"

# Create persistent model directory
mkdir -p /home/ubuntu/models

# Systemd Service Unit
sudo tee /etc/systemd/system/roadvision.service > /dev/null <<EOF
[Unit]
Description=RoadVision AI FastAPI Segmentation Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=$APP_DIR/backend
ExecStart=$VENV_DIR/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
Environment="PATH=$VENV_DIR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
Environment="MODEL_CHECKPOINT_URL=${MODEL_CHECKPOINT_URL:-https://huggingface.co/PalakChaudhary27/roadvision-deeplabv3-resnet50/resolve/main/best_model.pth}"
Environment="MODEL_CHECKPOINT_PATH=/home/ubuntu/models/best_model.pth"
Environment="ALLOWED_ORIGINS=*"

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable roadvision
sudo systemctl restart roadvision
echo "RoadVision systemd service installed and started!"
