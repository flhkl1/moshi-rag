#!/bin/bash
set -e

cd /workspace

# Clone repo if not already present
if [ ! -d "moshi-rag" ]; then
    git clone https://github.com/flhkl1/moshi-rag.git
fi

cd /workspace/moshi-rag

# Install Python package
pip install -e /workspace/moshi-rag/moshi

# Pin torch ecosystem to compatible versions
pip install torch==2.9.1 torchvision xformers==0.0.33.post2 --force-reinstall

# Fix editable install .pth file (hatchling bug — path is wrong by default)
echo "/workspace/moshi-rag" > /usr/local/lib/python3.11/dist-packages/_editable_impl_moshi.pth

# Install system dependencies
apt-get update -qq && apt-get install -y -qq nano
curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null
apt-get remove -y nodejs libnode-dev 2>/dev/null || true
apt-get install -y nodejs

# Build frontend
cd /workspace/moshi-rag/client
npm install && npm run build

echo "Setup complete. Run the following in separate terminals:"
echo ""
echo "Terminal 1 — Reference encoder:"
echo "  python3 -m moshi.moshi.server_conditioner \\"
echo "    --config hf://kyutai/moshika-rag-pytorch-bf16/config.json \\"
echo "    --moshi-weight hf://kyutai/moshika-rag-pytorch-bf16/model.safetensors \\"
echo "    --cuda-device 0 --conditioner reference_with_time --port 8001"
echo ""
echo "Terminal 2 — Main server:"
echo "  cd /workspace/moshi-rag && python3 -m moshi.moshi.server \\"
echo "    --gradio-tunnel --static ./client/dist --init-active-speaker model --gradium-stt"
