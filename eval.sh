#!/bin/bash

# Root directories
CONFIGS_ROOT="./configs"
WEIGHTS_ROOT="./RFUAV/weight" # Weights can be cloned or downloaded manually: git clone https://huggingface.co/datasets/kitofrank/RFUAV/
DATASET_ROOT="./dataset"
RES_ROOT="./res"

mkdir -p "$RES_ROOT"

# Array of config,weight relative paths
CONFIGS_WEIGHTS=(
"exp1.1_ResNet18.yaml exp1/ResNet18.pth"
"exp1.2_ResNet34.yaml exp1/ResNet34.pth"
"exp1.3_ResNet50.yaml exp1/ResNet50.pth"
"exp1.4_ResNet101.yaml exp1/ResNet101.pth"
"exp1.5_ResNet152.yaml exp1/ResNet152.pth"
"exp1.6_mobilenet_v3_s.yaml exp1/mobilenet_v3_small.pth"
"exp1.7_mobilenet_v3_l.yaml exp1/mobilenet_v3_large.pth"
"exp1.8_vit_b_16.yaml exp1/vit_b_16.pth"
"exp1.9_vit_b_32.yaml exp1/vit_b_32.pth"
"exp1.10_vit_l_16.yaml exp1/vit_l_16.pth"
"exp1.11_vit_l_32.yaml exp1/vit_l_32.pth"
"exp1.12_swin_v2_t.yaml exp1/swin_v2_t.pth"
"exp1.13_swin_v2_s.yaml exp1/swin_v2_s.pth"
"exp1.14_swin_v2_b.yaml exp1/swin_v2_b.pth"
"exp2.10_vit_l_16_autumn.yaml exp2/autumn/vit_l_16.pth"
"exp2.10_vit_l_16_hot.yaml exp2/hot/vit_l_16.pth"
"exp2.10_vit_l_16_hsv.yaml exp2/hsv/vit_l_16.pth"
"exp2.10_vit_l_16_parural.yaml exp2/parural/vit_l_16.pth"
)

for pair in "${CONFIGS_WEIGHTS[@]}"; do
    # Split into config and weight
    CONFIG_REL=$(echo $pair | awk '{print $1}')
    WEIGHT_REL=$(echo $pair | awk '{print $2}')

    CONFIG="$CONFIGS_ROOT/$CONFIG_REL"
    WEIGHT="$WEIGHTS_ROOT/$WEIGHT_REL"

    CONFIG_NAME=$(basename "$CONFIG" .yaml)
    OUT_DIR="$RES_ROOT/$CONFIG_NAME"
    mkdir -p "$OUT_DIR"

    for f in "$DATASET_ROOT"/*.iq; do
        FILE_NAME=$(basename "$f" .iq)
        FILE_DIR="$OUT_DIR/${FILE_NAME%.*}"
        mkdir -p "$FILE_DIR"

        # Log file per config (or per file if desired)
        LOG_FILE="$OUT_DIR/$FILE_NAME.log"

        python3 inference.py "$CONFIG" "$WEIGHT" "$f" "$FILE_DIR" &> "$LOG_FILE"
    done
done

python3 process_eval_log.py
