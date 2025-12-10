#!/bin/bash
# Helper script to train with different encoder backbones

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Multi-Task Ultrasound Model Training${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Default values
DATA_ROOT="${DATA_ROOT:-/root/baseline/train}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"

echo "Select encoder backbone:"
echo ""
echo "DINOv3 Models (Transformer-based):"
echo "  1) DINOv3 ViT-Small/16   (~2GB VRAM, Fast)"
echo "  2) DINOv3 ViT-Base/16    (~4GB VRAM, Balanced) ⭐ Recommended"
echo "  3) DINOv3 ViT-Base/14    (~4GB VRAM, Better quality)"
echo "  4) DINOv3 ViT-Large/14   (~8GB VRAM, Best quality)"
echo ""
echo "Traditional CNN Models:"
echo "  5) EfficientNet-B4       (~4GB VRAM, Fast)"
echo "  6) ResNet-34             (~3GB VRAM, Very fast)"
echo "  7) ResNet-50             (~4GB VRAM, Balanced)"
echo ""
read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        ENCODER="facebook/dinov3-vits16-pretrain-lvd1689m"
        WEIGHTS=""
        echo -e "${GREEN}Selected: DINOv3 ViT-Small/16${NC}"
        ;;
    2)
        ENCODER="facebook/dinov3-vitb16-pretrain-lvd1689m"
        WEIGHTS=""
        echo -e "${GREEN}Selected: DINOv3 ViT-Base/16${NC}"
        ;;
    3)
        ENCODER="facebook/dinov3-vitb14-pretrain-lvd1689m"
        WEIGHTS=""
        echo -e "${GREEN}Selected: DINOv3 ViT-Base/14${NC}"
        ;;
    4)
        ENCODER="facebook/dinov3-vitl14-pretrain-lvd1689m"
        WEIGHTS=""
        echo -e "${GREEN}Selected: DINOv3 ViT-Large/14${NC}"
        ;;
    5)
        ENCODER="efficientnet-b4"
        WEIGHTS="--encoder_weights imagenet"
        echo -e "${GREEN}Selected: EfficientNet-B4${NC}"
        ;;
    6)
        ENCODER="resnet34"
        WEIGHTS="--encoder_weights imagenet"
        echo -e "${GREEN}Selected: ResNet-34${NC}"
        ;;
    7)
        ENCODER="resnet50"
        WEIGHTS="--encoder_weights imagenet"
        echo -e "${GREEN}Selected: ResNet-50${NC}"
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Encoder:    $ENCODER"
echo "  Data root:  $DATA_ROOT"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs:     $NUM_EPOCHS"
echo ""

read -p "Proceed with training? (y/n): " confirm

if [[ $confirm != [yY] ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo -e "${GREEN}Starting training...${NC}"
echo ""

# Construct command
CMD="python train.py \
    --encoder_name $ENCODER \
    $WEIGHTS \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --data_root $DATA_ROOT"

echo "Running: $CMD"
echo ""

# Run training
eval $CMD

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Training completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Model saved as: best_model.pth"
echo ""
echo "To run inference:"
echo "  python model.py --encoder_name $ENCODER --model_path best_model.pth"
