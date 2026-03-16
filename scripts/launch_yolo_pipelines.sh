#!/usr/bin/env bash
# ============================================================================
# launch_yolo_pipelines.sh
# Launches 3 YOLO overlay pipelines in a tmux session
# ============================================================================
# Usage:
#   bash launch_yolo_pipelines.sh              # default: front=0, side=2, wrist=4
#   bash launch_yolo_pipelines.sh 0 2 4        # custom device indices
#
# Prerequisites:
#   - setup_virtual_cameras.sh already run
#   - tmux installed (sudo apt install tmux)
#   - conda env 'lerobot_lab' exists with ultralytics + pyvirtualcam
# ============================================================================
set -euo pipefail

FRONT_SRC="${1:-/dev/video0}"
SIDE_SRC="${2:-/dev/video2}"
WRIST_SRC="${3:-/dev/video4}"

SESSION="sevo_yolo"
CONDA_ENV="lerobot_lab"
WORK_DIR="$HOME/lerobot_lab"
SCRIPT="tools/yolo/yolo_seg_highlight_to_v4l2.py"

COMMON_ARGS="--w 640 --h 360 --fps 260 --model yolov8n-seg.pt --infer_fps 15 --conf 0.20 --iou 0.5 --mask_th 0.3 --alpha 0.45"

# Kill existing session if any
tmux kill-session -t $SESSION 2>/dev/null || true

echo "[SEVO] Launching YOLO pipelines in tmux session '$SESSION'..."

# Create session with front camera
tmux new-session -d -s $SESSION -n "front" \
  "cd $WORK_DIR && conda activate $CONDA_ENV && python $SCRIPT --src $FRONT_SRC --out /dev/video10 $COMMON_ARGS; read"

# Add side camera pane
tmux new-window -t $SESSION -n "side" \
  "cd $WORK_DIR && conda activate $CONDA_ENV && python $SCRIPT --src $SIDE_SRC --out /dev/video11 $COMMON_ARGS; read"

# Add wrist camera pane
tmux new-window -t $SESSION -n "wrist" \
  "cd $WORK_DIR && conda activate $CONDA_ENV && python $SCRIPT --src $WRIST_SRC --out /dev/video12 $COMMON_ARGS; read"

echo "[SEVO] 3 YOLO pipelines launched."
echo "  Front: $FRONT_SRC → /dev/video10"
echo "  Side:  $SIDE_SRC → /dev/video11"
echo "  Wrist: $WRIST_SRC → /dev/video12"
echo ""
echo "Attach with: tmux attach -t $SESSION"
echo "Stop all:    tmux kill-session -t $SESSION"
