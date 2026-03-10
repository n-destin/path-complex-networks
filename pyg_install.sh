#!/usr/bin/env bash
TORCH=1.12.1
CUDA=$1  # Supply as command line cpu or cu102
pip install --no-cache-dir --force-reinstall torch-scatter==2.1.0 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-cache-dir --force-reinstall torch-sparse==0.6.16 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-cache-dir --force-reinstall torch-cluster==1.6.0 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-cache-dir --force-reinstall torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-cache-dir --force-reinstall torch-geometric==2.4.0