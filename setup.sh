#!/bin/bash

DONE=/workspace/.setup_done  # checkpoint dir on persistent volume

mkdir -p $DONE

cd /workspace

# Clone repo if not already present
if [ ! -d "moshi-rag" ]; then
    git clone https://github.com/flhkl1/moshi-rag.git
fi

cd /workspace/moshi-rag

# Point HuggingFace cache to persistent volume so models survive restarts
export HF_HOME=/workspace/.cache/huggingface
mkdir -p $HF_HOME

# HuggingFace login (token set as RunPod env var HUGGING_FACE_HUB_TOKEN)
if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
    huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN"
fi

# Install Python package + torch (skip if already done)
if [ ! -f $DONE/pip ]; then
    pip install -e /workspace/moshi-rag/moshi
    python3 -c "import torch; assert torch.__version__.startswith('2.9.1')" 2>/dev/null || \
        pip install torch==2.9.1 torchvision xformers==0.0.33.post2 --force-reinstall
    touch $DONE/pip
fi

# Fix editable install .pth file (hatchling bug — path is wrong by default)
echo "/workspace/moshi-rag" > /usr/local/lib/python3.11/dist-packages/_editable_impl_moshi.pth

# Install system dependencies (skip if already done)
if [ ! -f $DONE/apt ]; then
    apt-get update -qq && apt-get install -y -qq nano
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null
    apt-get remove -y nodejs libnode-dev 2>/dev/null || true
    apt-get install -y nodejs
    touch $DONE/apt
fi

# Build frontend (skip if already done)
if [ ! -f $DONE/npm ]; then
    cd /workspace/moshi-rag/client
    npm install && npm run build
    touch $DONE/npm
fi

echo "Setup complete. Starting servers..."

# Kill anything on port 8001 (nginx occupies it by default)
fuser -k 8001/tcp 2>/dev/null || true

# Start reference encoder in background
cd /workspace/moshi-rag
python3 -m moshi.moshi.server_conditioner \
    --config hf://kyutai/moshika-rag-pytorch-bf16/config.json \
    --moshi-weight hf://kyutai/moshika-rag-pytorch-bf16/model.safetensors \
    --cuda-device 0 --conditioner reference_with_time --port 8001 &

# Wait for port 8001 to be ready
echo "Waiting for reference encoder on port 8001..."
until ss -tlnp | grep -q ':8001'; do sleep 3; done
echo "Reference encoder ready. Starting main server..."

# Start main server in foreground (keeps container alive, prints gradio tunnel URL)
python3 -m moshi.moshi.server \
    --gradio-tunnel --static ./client/dist --init-active-speaker model --gradium-stt
