#!/usr/bin/env bash
# ============================================================================
# setup_virtual_cameras.sh
# Run after every reboot to create SEVO virtual camera devices
# ============================================================================
set -euo pipefail

echo "[SEVO] Creating 3 v4l2loopback virtual cameras..."
sudo modprobe v4l2loopback devices=3 video_nr=10,11,12 \
  card_label="YOLO-front","YOLO-side","YOLO-wrist" exclusive_caps=0

echo "[SEVO] Setting permissions..."
sudo chmod 666 /dev/video10 /dev/video11 /dev/video12

echo "[SEVO] Setting fps cap to 260..."
echo '@260' | sudo tee /sys/devices/virtual/video4linux/video10/format > /dev/null
echo '@260' | sudo tee /sys/devices/virtual/video4linux/video11/format > /dev/null
echo '@260' | sudo tee /sys/devices/virtual/video4linux/video12/format > /dev/null

echo "[SEVO] Verifying..."
v4l2-ctl --list-devices 2>/dev/null | grep -A1 YOLO || echo "WARNING: YOLO devices not found"

echo ""
echo "[SEVO] Virtual cameras ready:"
echo "  /dev/video10 (YOLO-front)"
echo "  /dev/video11 (YOLO-side)"
echo "  /dev/video12 (YOLO-wrist)"
echo ""
echo "Next: launch YOLO pipelines with launch_yolo_pipelines.sh"
