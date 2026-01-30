#!/bin/bash
# Setup script for mlx-bitnet testing with JARVIS

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "======================================"
echo "Setting up mlx-bitnet for JARVIS"
echo "======================================"
echo "Project directory: $PROJECT_DIR"

# 1. Clone mlx-bitnet to the project directory
cd "$PROJECT_DIR"
if [ ! -d "mlx-bitnet" ]; then
    echo "📦 Cloning mlx-bitnet repository..."
    git clone https://github.com/exo-explore/mlx-bitnet.git
fi

cd mlx-bitnet

# 2. Install dependencies
echo "📦 Installing dependencies..."
uv pip install -r requirements.txt

# 3. Download and convert model
echo "📥 Downloading BitNet model (this may take a few minutes)..."
uv run python convert.py

echo ""
echo "======================================"
echo "Setup complete!"
echo "======================================"
echo ""
echo "To test the model:"
echo "  cd $PROJECT_DIR/mlx-bitnet"
echo "  uv run python test_interop.py"
echo ""
echo "To run inference:"
echo "  uv run python mlx_bitnet.py --prompt 'Hello, how are you?'"
echo ""
echo "Note: The model will be downloaded to ~/.cache/huggingface/"
echo "Model size: ~0.4GB (2B parameters at 1.58-bit)"
