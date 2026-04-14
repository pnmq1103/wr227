#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

PROJECT_DIR="$HOME/workspace"
DATA_DIR="$HOME/data/imagenet-100"
NUM_GPUS=2 # <-- Change this to match rented server's GPU count

echo "Setting up workspace in $PROJECT_DIR..."
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

if [ ! -d "RALA" ]; then
  echo "Cloning RALA repository..."
  git clone https://github.com/qhfan/RALA.git
else
  echo "RALA repository already exists. Skipping clone."
fi

echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Kaggle API alongside other dependencies
cd RALA/classfication
echo "Installing dependencies..."
pip install fvcore "timm==0.9.10" scipy einops scikit-learn matplotlib kaggle
echo "Applying weights_only patch to main.py..."
sed -i "s/map_location='cpu')/map_location='cpu', weights_only=False)/g" main.py

echo "Preparing Data Directory..."
mkdir -p "$DATA_DIR"
if [ ! -d "$DATA_DIR/train.X1" ] && [ ! -d "$DATA_DIR/train" ]; then
  echo "Downloading ImageNet-100 from Kaggle..."
  kaggle datasets download -d ambityga/imagenet100 -p "$DATA_DIR" --unzip
else
  echo "Dataset folders found. Skipping download."
fi

echo "Merging ImageNet-100 train..."
mkdir -p "$DATA_DIR/train"
mv "$DATA_DIR"/train.X*/* "$DATA_DIR/train/" 2>/dev/null || true
rmdir "$DATA_DIR"/train.X* 2>/dev/null || true

echo "Merging ImageNet-100 val..."
mkdir -p "$DATA_DIR/val"
mv "$DATA_DIR"/val.X*/* "$DATA_DIR/val/" 2>/dev/null || true
rmdir "$DATA_DIR"/val.X* 2>/dev/null || true

# ==========================================
# TRAINING
# ==========================================
echo "Starting distributed training on $NUM_GPUS GPUs..."
mkdir -p "$PROJECT_DIR/RALA_Output"

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS main.py \
  --warmup-epochs 1 \
  --model RAVLT_T \
  --data-path "$DATA_DIR" \
  --num_workers 8 \
  --batch-size 128 \
  --lr 1e-3 \
  --epochs 40 \
  --dist-eval \
  --output_dir "$PROJECT_DIR/RALA_Output"

echo "Training complete! Outputs saved to $PROJECT_DIR/RALA_Output"
