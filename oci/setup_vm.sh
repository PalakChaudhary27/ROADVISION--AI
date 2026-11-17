#!/usr/bin/env bash
# ==============================================================================
# RoadVision AI — Automated OCI Ampere A1 Setup & Provisioning Script
# Target: Ubuntu 22.04 LTS (ARM64 / aarch64) on Oracle Cloud Always Free
# ==============================================================================
set -euo pipefail

echo "=========================================================="
echo " Starting RoadVision AI Setup on Oracle Cloud ARM64..."
echo "=========================================================="

# 1. Update OS and install prerequisites
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    gnupg \
    lsb-release \
    iptables-persistent \
    net-tools \
    htop

# 2. Configure Ubuntu firewall (iptables) to allow inbound HTTP/HTTPS/8000
echo "--> Configuring firewall rules..."
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT || true
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT || true
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT || true
sudo netfilter-persistent save || true

# 3. Install Docker Engine & Docker Compose for ARM64 if not present
if ! command -v docker &> /dev/null; then
    echo "--> Installing Docker Engine for ARM64..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
    sudo usermod -aG docker "$USER"
    sudo systemctl enable docker
    sudo systemctl start docker
fi

# 4. Clone or pull repository
PROJECT_DIR="/home/ubuntu/RoadVision-AI"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "--> Cloning RoadVision AI repository..."
    git clone https://github.com/PalakChaudhary27/ROADVISION--AI.git "$PROJECT_DIR"
else
    echo "--> Updating existing repository..."
    cd "$PROJECT_DIR" && git pull origin main
fi

cd "$PROJECT_DIR"

# 5. Create persistent models directory and pre-download checkpoint if URL provided
MODEL_URL="${MODEL_CHECKPOINT_URL:-}"
mkdir -p /home/ubuntu/roadvision_models
if [ -n "$MODEL_URL" ] && [ ! -f /home/ubuntu/roadvision_models/best_model.pth ]; then
    echo "--> Pre-downloading model checkpoint from Hugging Face..."
    curl -L -o /home/ubuntu/roadvision_models/best_model.pth "$MODEL_URL"
fi

# 6. Build and launch Docker container
echo "--> Building and starting RoadVision backend container..."
sudo docker compose down || true
sudo docker compose up -d --build

# 7. Verification check
echo "--> Waiting for container to boot and load model..."
sleep 15
curl -s http://localhost:8000/health | grep -q "healthy" && echo " SUCCESS: RoadVision AI Backend is Healthy!" || echo " Checking logs: sudo docker compose logs"

echo "=========================================================="
echo " Setup complete! RoadVision API running on port 8000"
echo "=========================================================="
