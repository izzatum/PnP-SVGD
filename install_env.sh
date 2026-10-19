#!/bin/bash
# 
# Installer for PnPSVGD environment
# 
# Run: ./install_env.sh
# 
# M. Izzatullah, 15/12/2022

# download DRUNet model
echo 'Downloading required DRUNet model'

FILEID='1g3UcCEjVsnziCwtRyBZKGz9HlCsMsWmU'
FILENAME=$(pwd)/models/drunet_gray.pth

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $FILENAME && rm -rf /tmp/cookies.txt

echo 'Download complete!'

# create conda env
echo 'Creating pnpsvgd environment'

conda env create -f environment.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pnpsvgd
conda env list
echo 'Created and activated environment:' $(which python)

# Check cupy works as expected
echo 'Checking cupy version and running a command...'
python -c 'import cupy as cp; print(cp.__version__); import torch; print(torch.__version__);  print(torch.cuda.get_device_name(torch.cuda.current_device())); print(torch.ones(10).to("cuda:0"))'

echo 'Done!'
