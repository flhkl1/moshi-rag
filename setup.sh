#!/bin/bash

DONE=/workspace/.setup_done  # checkpoint dir on persistent volume
log() { echo "[setup] $(date '+%H:%M:%S') $*"; }

log "Starting setup..."
mkdir -p $DONE

cd /workspace

# Clone repo if not already present
if [ ! -d "moshi-rag" ]; then
    log "Cloning moshi-rag repo..."
    git clone https://github.com/flhkl1/moshi-rag.git
    log "Repo cloned."
else
    log "Repo already present, skipping clone."
fi

cd /workspace/moshi-rag

# Point HuggingFace cache to persistent volume so models survive restarts
export HF_HOME=/workspace/.cache/huggingface
mkdir -p $HF_HOME
log "HuggingFace cache → $HF_HOME"

# HuggingFace login
if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
    log "Logging into HuggingFace..."
    huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN"
    log "HuggingFace login done."
else
    log "WARNING: HUGGING_FACE_HUB_TOKEN not set — gated models will fail."
fi

# Install Python package + torch
if [ ! -f $DONE/pip ]; then
    log "Installing Python dependencies (this takes ~3 min)..."
    pip install -e /workspace/moshi-rag/moshi
    log "Checking torch version..."
    python3 -c "import torch; assert torch.__version__.startswith('2.9.1')" 2>/dev/null \
        && log "torch 2.9.1 already installed, skipping reinstall." \
        || { log "Pinning torch==2.9.1 + torchvision + xformers..."; pip install torch==2.9.1 torchvision xformers==0.0.33.post2 --force-reinstall; }
    touch $DONE/pip
    log "Python dependencies done."
else
    log "Python dependencies already installed, skipping."
fi

# Fix editable install .pth file (hatchling bug — path is wrong by default)
log "Fixing .pth file..."
echo "/workspace/moshi-rag" > /usr/local/lib/python3.11/dist-packages/_editable_impl_moshi.pth

# Install system dependencies
if [ ! -f $DONE/apt ]; then
    log "Installing system dependencies (nano, Node 20)..."
    apt-get update -qq && apt-get install -y -qq nano
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null
    apt-get remove -y nodejs libnode-dev 2>/dev/null || true
    apt-get install -y nodejs
    touch $DONE/apt
    log "System dependencies done."
else
    log "System dependencies already installed, skipping."
fi

# Build frontend
if [ ! -f $DONE/npm ]; then
    log "Building frontend..."
    cd /workspace/moshi-rag/client
    npm install && npm run build
    touch $DONE/npm
    log "Frontend built."
else
    log "Frontend already built, skipping."
fi

log "Setup complete. Starting servers..."

# Kill anything on port 8001 (nginx occupies it by default)
log "Clearing port 8001..."
fuser -k 8001/tcp 2>/dev/null || true

# Start reference encoder in background
log "Starting reference encoder on port 8001..."
cd /workspace/moshi-rag
python3 -m moshi.moshi.server_conditioner \
    --config hf://kyutai/moshika-rag-pytorch-bf16/config.json \
    --moshi-weight hf://kyutai/moshika-rag-pytorch-bf16/model.safetensors \
    --cuda-device 0 --conditioner reference_with_time --port 8001 &

log "Waiting for reference encoder on port 8001 (model loading takes 1-2 min)..."
until ss -tlnp | grep -q ':8001'; do sleep 3; done
log "Reference encoder ready."

log "Starting main Moshi server — Gradio tunnel URL will appear below..."
python3 -m moshi.moshi.server \
    --gradio-tunnel --static ./client/dist --init-active-speaker model --gradium-stt
