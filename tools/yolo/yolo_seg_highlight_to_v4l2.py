#!/usr/bin/env python3
"""
SEVO Virtual Camera: YOLOv8-seg overlay → v4l2loopback

Reads a physical USB camera, runs YOLOv8 instance segmentation at a reduced
frame rate, blends a colored overlay onto detected object masks, and writes
the result to a v4l2loopback virtual camera device.

The virtual camera output can then be consumed by LeRobot (or any V4L2 client)
as if it were a normal USB camera.

Usage:
    python yolo_seg_highlight_to_v4l2.py \
        --src /dev/video0 --out /dev/video10 \
        --w 640 --h 360 --fps 260 \
        --model yolov8n-seg.pt --infer_fps 15 \
        --conf 0.20 --iou 0.5 --mask_th 0.3 \
        --alpha 0.45

Overlay formula (per pixel):
    Ĩ_t = (1 - α·M_t) ⊙ I_t + α·M_t ⊙ C

where:
    I_t   = raw RGB frame
    M_t   = binary segmentation mask from YOLO (0 or 1)
    α     = blending strength (default 0.45)
    C     = overlay color in BGR (default [255, 255, 0] = yellow)
    Ĩ_t   = SEVO-enhanced output frame
"""

import argparse
import time
import cv2
import numpy as np
import pyvirtualcam
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(
        description="SEVO: YOLO-seg overlay → v4l2loopback virtual camera"
    )
    # Camera I/O
    p.add_argument("--src", required=True,
                   help="Source camera (e.g. /dev/video0 or integer index)")
    p.add_argument("--out", required=True,
                   help="Output v4l2loopback device (e.g. /dev/video10)")
    p.add_argument("--w", type=int, default=640, help="Frame width")
    p.add_argument("--h", type=int, default=360, help="Frame height")
    p.add_argument("--fps", type=int, default=260,
                   help="Virtual camera fps cap (match v4l2loopback format setting)")

    # YOLO config
    p.add_argument("--model", default="yolov8n-seg.pt",
                   help="YOLOv8 segmentation model path")
    p.add_argument("--infer_fps", type=int, default=15,
                   help="YOLO inference rate (run detection every 1/infer_fps seconds)")
    p.add_argument("--conf", type=float, default=0.20,
                   help="YOLO confidence threshold")
    p.add_argument("--iou", type=float, default=0.5,
                   help="YOLO NMS IoU threshold")
    p.add_argument("--mask_th", type=float, default=0.3,
                   help="Mask binarization threshold")
    p.add_argument("--classes", nargs="+", type=int, default=[39],
                   help="COCO class IDs to detect (39=bottle, 41=cup, 64=mouse)")

    # Overlay config
    p.add_argument("--alpha", type=float, default=0.45,
                   help="Overlay blending strength (0=transparent, 1=opaque)")
    p.add_argument("--color", nargs=3, type=int, default=[255, 255, 0],
                   help="Overlay color in BGR (default: [255,255,0] = yellow)")

    return p.parse_args()


def open_source_camera(src, width, height):
    """Open physical camera with MJPG fourcc."""
    if isinstance(src, str) and src.startswith("/dev/video"):
        idx = int(src.replace("/dev/video", ""))
    else:
        try:
            idx = int(src)
        except ValueError:
            idx = src

    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source camera: {src} (index={idx})")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[SEVO] Source camera opened: {src} ({actual_w}x{actual_h})")
    return cap


def main():
    args = parse_args()

    # Open physical camera
    cap = open_source_camera(args.src, args.w, args.h)

    # Load YOLO model
    model = YOLO(args.model)
    print(f"[SEVO] Model loaded: {args.model}")
    print(f"[SEVO] Detecting classes: {args.classes} (39=bottle)")
    print(f"[SEVO] Conf={args.conf}, IoU={args.iou}, mask_th={args.mask_th}")
    print(f"[SEVO] Overlay: alpha={args.alpha}, color(BGR)={args.color}")

    # Open virtual camera
    cam = pyvirtualcam.Camera(
        width=args.w, height=args.h, fps=args.fps,
        device=args.out, fmt=pyvirtualcam.PixelFormat.BGR
    )
    print(f"[SEVO] Virtual camera: {args.out} ({args.w}x{args.h}@{args.fps}fps)")
    print(f"[SEVO] Ready. Press Ctrl+C to stop.\n")

    overlay_color = np.array(args.color, dtype=np.float32)
    infer_interval = 1.0 / args.infer_fps
    last_infer = 0.0
    current_mask = None
    frame_count = 0
    detect_count = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, (args.w, args.h))
            now = time.time()
            frame_count += 1

            # Run YOLO at reduced rate to save GPU
            if now - last_infer >= infer_interval:
                results = model(
                    frame, conf=args.conf, iou=args.iou,
                    classes=args.classes, verbose=False
                )
                last_infer = now

                if results[0].masks is not None and len(results[0].masks.data) > 0:
                    masks = results[0].masks.data.cpu().numpy()
                    combined = np.any(masks > args.mask_th, axis=0).astype(np.float32)
                    current_mask = cv2.resize(combined, (args.w, args.h))
                    detect_count += 1
                else:
                    current_mask = None

            # Apply per-pixel overlay
            if current_mask is not None:
                m = current_mask[:, :, np.newaxis]
                output = (
                    (1.0 - args.alpha * m) * frame.astype(np.float32)
                    + args.alpha * m * overlay_color
                )
                output = np.clip(output, 0, 255).astype(np.uint8)
            else:
                output = frame

            cam.send(output)
            cam.sleep_until_next_frame()

            # Print stats every 5 seconds
            elapsed = now - t_start
            if elapsed > 0 and frame_count % 150 == 0:
                fps_actual = frame_count / elapsed
                print(
                    f"[SEVO] frames={frame_count}, "
                    f"fps={fps_actual:.1f}, "
                    f"detections={detect_count}",
                    end="\r"
                )

    except KeyboardInterrupt:
        elapsed = time.time() - t_start
        print(f"\n[SEVO] Stopped. {frame_count} frames in {elapsed:.1f}s "
              f"({frame_count/elapsed:.1f} fps avg)")
    finally:
        cap.release()
        cam.close()


if __name__ == "__main__":
    main()
