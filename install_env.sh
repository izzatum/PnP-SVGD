#!/usr/bin/env bash
# Create the pnpsvgd conda environment, install the package and fetch the DRUNet weights.
#
# Usage: ./install_env.sh   (safe to run again: it updates the environment if it exists)
set -euo pipefail

cd "$(dirname "$0")"

WEIGHTS=models/drunet_gray.pth
WEIGHTS_URL=https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth
WEIGHTS_SHA256=e27fb29456732c604c8ee3ac3e92ecadd2a7a4e36a8be675e92ebbf93a240de4

echo "Creating the pnpsvgd conda environment"
conda env create -f environment.yml || conda env update -f environment.yml
eval "$(conda shell.bash hook)"
conda activate pnpsvgd
pip install -e .

sha256() { python -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"; }

if [ ! -f "$WEIGHTS" ] || [ "$(sha256 "$WEIGHTS")" != "$WEIGHTS_SHA256" ]; then
  echo "Downloading the DRUNet weights to $WEIGHTS"
  curl -fL --retry 3 -o "$WEIGHTS.part" "$WEIGHTS_URL"
  if [ "$(sha256 "$WEIGHTS.part")" != "$WEIGHTS_SHA256" ]; then
    rm -f "$WEIGHTS.part"
    echo "The downloaded weights do not match the expected SHA-256. Nothing was installed." >&2
    exit 1
  fi
  mv "$WEIGHTS.part" "$WEIGHTS"
fi

echo "Checking PyTorch and the GPU"
python -c 'import torch; print("torch", torch.__version__); print("GPU:", torch.cuda.get_device_name(0))'

echo "Done. Activate the environment with: conda activate pnpsvgd"
