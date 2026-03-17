# SEVO: Semantic-Enhanced Virtual Observation

> A data-centric observation pipeline for improving cross-environment robustness of imitation-learning policies on low-cost robot hardware.
>
> **Scope:** Single-task pick-and-place with transparent bottles. This is a case study in observation-space design, not a general-purpose manipulation system.

## What This Is (and What It Is Not)

SEVO is a method for modifying **what the policy sees**, not how it learns. It combines YOLO segmentation overlay, active red LED illumination, and diversified data collection to help standard policies (ACT, SmolVLA) transfer across visually different environments.

**What SEVO demonstrates:**
- Observation-space modifications (overlay + lighting + data diversity) can substantially improve cross-environment transfer for a single manipulation task
- Diversified backgrounds during data collection are the dominant factor for generalization, outranking both active illumination and YOLO overlay
- Body-fixed cameras substantially outperform wrist cameras on low-cost SO-101 arms
- These effects are consistent across two robot platforms, two policy architectures, and hundreds of real-robot trials

**What SEVO does NOT address:**
- Multi-task manipulation (e.g., BEHAVIOR benchmark tasks)
- Multi-object discrimination (the overlay applies the same color to all detected objects; the policy cannot distinguish "pick bottle A, not cup B" through the overlay alone)
- Language-conditioned task switching (SmolVLA receives a fixed prompt; no dynamic instruction following is evaluated)
- Generalization to object categories not in the YOLO training set
- Comparison with large-scale VLAs (π0.5, OpenVLA, GR00T) that target general-purpose manipulation

This work targets a specific, well-defined problem: closing the background-generalization gap for a single task on community-accessible hardware. Extending to multi-task settings requires additional mechanisms (e.g., per-class overlay colors, language-conditioned detection, broader task coverage in training data) that are not validated here.

## Results

Evaluated over 80-100 real-robot trials per condition.

| Setting | ACT | SmolVLA |
|---------|-----|---------|
| Training environment | 95% | 83% |
| Novel environments (similar floor tone) | 85% | 75% |
| Novel environments (extreme, out-of-distribution floor) | 10% | 5% |
| **No SEVO, training env** | 75% | 70% |
| **No SEVO, novel env** | 30% | 35% |
| **No SEVO, extreme** | 0% | 0% |

The extreme-environment gap (bright white floor absent from all training data) confirms that SEVO's robustness is bounded by the visual diversity of the training distribution. A separate training run in the white-floor lab closed this gap, indicating the limitation is data coverage, not the method itself.

**Component importance ranking (from ablation):**
```
Varied Backgrounds (dominant) > Red LED (second) > YOLO Overlay (third)
```

---

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Software Setup (One-Time)](#software-setup-one-time)
- [Boot to Evaluation (5 Phases)](#step-by-step-boot-to-evaluation)
- [Data Collection Protocol](#data-collection-protocol)
- [Architecture Details](#architecture-details)
- [Known Limitations](#known-limitations)

---

## Hardware Requirements

### Per Robot
- **Arm:** SO-101 leader + follower pair
- **Cameras:** 2-3x USB cameras (640x360, MJPG capable)
- **LED:** 5W red LED panel (620-630nm dominant wavelength)
- **Compute (onboard):** Jetson Orin NX 16GB or Raspberry Pi 5 8GB

### Training Server
- NVIDIA GPU with 24GB+ VRAM (RTX 4090 recommended)
- 32GB+ RAM, 100GB+ storage for datasets

---

## Software Setup (One-Time)

```bash
# 1. Clone and install LeRobot
git clone https://github.com/huggingface/lerobot.git ~/lerobot_lab
cd ~/lerobot_lab
conda create -n lerobot_lab python=3.10 -y
conda activate lerobot_lab
pip install -e ".[act,smolvla]"

# 2. Install v4l2loopback (Linux virtual camera kernel module)
sudo apt install -y v4l2loopback-dkms v4l2loopback-utils
modinfo v4l2loopback | head -3   # verify

# 3. Install YOLO dependencies
pip install ultralytics pyvirtualcam opencv-python-headless

# 4. Pre-download YOLOv8-seg model
python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"
```

---

## Step-by-Step: Boot to Evaluation

### Phase 1: Create Virtual Camera Devices (every reboot)

```bash
sudo modprobe v4l2loopback devices=3 video_nr=10,11,12 \
  card_label="YOLO-front","YOLO-side","YOLO-wrist" exclusive_caps=0

sudo chmod 666 /dev/video10 /dev/video11 /dev/video12

# Set high fps cap to prevent frame drops
echo '@260' | sudo tee /sys/devices/virtual/video4linux/video10/format
echo '@260' | sudo tee /sys/devices/virtual/video4linux/video11/format
echo '@260' | sudo tee /sys/devices/virtual/video4linux/video12/format

# Verify
v4l2-ctl --list-devices | grep -A1 YOLO
```

### Phase 2: Launch YOLO Overlay Pipelines (3 terminals)

> Adjust `--src` to match your physical camera device numbers.
> Check mapping with: `v4l2-ctl --list-devices`

**Terminal 1 (front):**
```bash
conda activate lerobot_lab && cd ~/lerobot_lab
python tools/yolo/yolo_seg_highlight_to_v4l2.py \
  --src /dev/video0 --out /dev/video10 \
  --w 640 --h 360 --fps 260 \
  --model yolov8n-seg.pt --infer_fps 15 \
  --conf 0.20 --iou 0.5 --mask_th 0.3 --alpha 0.45
```

**Terminal 2 (side):**
```bash
conda activate lerobot_lab && cd ~/lerobot_lab
python tools/yolo/yolo_seg_highlight_to_v4l2.py \
  --src /dev/video2 --out /dev/video11 \
  --w 640 --h 360 --fps 260 \
  --model yolov8n-seg.pt --infer_fps 15 \
  --conf 0.20 --iou 0.5 --mask_th 0.3 --alpha 0.45
```

**Terminal 3 (wrist, for ablation only):**
```bash
conda activate lerobot_lab && cd ~/lerobot_lab
python tools/yolo/yolo_seg_highlight_to_v4l2.py \
  --src /dev/video4 --out /dev/video12 \
  --w 640 --h 360 --fps 260 \
  --model yolov8n-seg.pt --infer_fps 15 \
  --conf 0.20 --iou 0.5 --mask_th 0.3 --alpha 0.45
```

**Verify:** `ffplay /dev/video10` should show the front camera with yellow bottle overlay.

### Phase 3: Teleoperate & Collect Data

#### SEVO dataset (YOLO overlay + red LED ON + diversified backgrounds)

```bash
conda activate lerobot_lab && cd ~/lerobot_lab
# Turn on red LED hardware before starting!

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{"front":{"type":"opencv","index_or_path":"/dev/video10","width":640,"height":360,"fps":260,"fourcc":"MJPG"},"side":{"type":"opencv","index_or_path":"/dev/video11","width":640,"height":360,"fps":260,"fourcc":"MJPG"},"wrist":{"type":"opencv","index_or_path":"/dev/video12","width":640,"height":360,"fps":260,"fourcc":"MJPG"}}' \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/sevo_yolo_redlight_bottle_pickup \
  --dataset.single_task="pick up the bottle and place it on the side" \
  --dataset.num_episodes=80 \
  --dataset.episode_time_s=1800 \
  --dataset.reset_time_s=10 \
  --dataset.video=true \
  --dataset.fps=60 \
  --dataset.push_to_hub=false
```

#### Baseline dataset (NO YOLO, NO red LED, physical cameras)

```bash
# Turn OFF red LED, use physical cameras (0/2/4 not 10/11/12)

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{"front":{"type":"opencv","index_or_path":0,"width":640,"height":360,"fps":260,"fourcc":"MJPG"},"side":{"type":"opencv","index_or_path":2,"width":640,"height":360,"fps":260,"fourcc":"MJPG"},"wrist":{"type":"opencv","index_or_path":4,"width":640,"height":360,"fps":260,"fourcc":"MJPG"}}' \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/baseline_noyolo_noredlight \
  --dataset.single_task="pick up the bottle and place it on the side" \
  --dataset.num_episodes=80 \
  --dataset.episode_time_s=1800 \
  --dataset.reset_time_s=10 \
  --dataset.video=true \
  --dataset.fps=60 \
  --dataset.push_to_hub=false
```

### Phase 4: Train

#### ACT (batch=24, ~400K steps)

```bash
conda activate lerobot_lab && cd ~/lerobot_lab

lerobot-train \
  --dataset.repo_id=${HF_USER}/sevo_yolo_redlight_bottle_pickup \
  --policy.type=act \
  --output_dir=outputs/train/act_b24_s400k_sevo_full \
  --job_name=act_sevo_full \
  --training.batch_size=24 \
  --training.steps=400000 \
  --policy.device=cuda \
  --wandb.enable=true
```

#### SmolVLA (batch=64, ~20K steps, fine-tune from pretrained)

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=${HF_USER}/sevo_yolo_redlight_bottle_pickup \
  --output_dir=outputs/train/smolvla_b64_s20k_sevo_full \
  --job_name=smolvla_sevo_full \
  --training.batch_size=64 \
  --training.steps=20000 \
  --policy.device=cuda \
  --wandb.enable=true
```

### Phase 5: Evaluate

#### Evaluate ACT with SEVO (ensure YOLO pipelines running + red LED ON)

```bash
conda activate lerobot_lab && cd ~/lerobot_lab

MODEL="act_b24_s400k_sevo_full"
CKPT="last"
POLICY="$PWD/outputs/train/${MODEL}/checkpoints/${CKPT}/pretrained_model"
EVAL_ID="eval_${MODEL}_ckpt${CKPT}"

rm -rf ~/.cache/huggingface/lerobot/${HF_USER}/${EVAL_ID}

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{"front":{"type":"opencv","index_or_path":"/dev/video10","width":640,"height":360,"fps":260,"fourcc":"MJPG"},"side":{"type":"opencv","index_or_path":"/dev/video11","width":640,"height":360,"fps":260,"fourcc":"MJPG"},"wrist":{"type":"opencv","index_or_path":"/dev/video12","width":640,"height":360,"fps":260,"fourcc":"MJPG"}}' \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/${EVAL_ID} \
  --dataset.single_task="Pick the bottle (eval ${MODEL} ckpt${CKPT})" \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=1800 \
  --dataset.reset_time_s=10 \
  --dataset.video=true \
  --dataset.fps=60 \
  --dataset.push_to_hub=false \
  --policy.path=${POLICY} \
  --robot.disable_torque_on_disconnect=false
```

#### Evaluate SmolVLA with SEVO

```bash
# NOTE: SmolVLA uses camera keys camera1/camera2/camera3

MODEL="smolvla_b64_s20k_sevo_full"
CKPT="last"
POLICY="$PWD/outputs/train/${MODEL}/checkpoints/${CKPT}/pretrained_model"
EVAL_ID="eval_${MODEL}_ckpt${CKPT}"

rm -rf ~/.cache/huggingface/lerobot/${HF_USER}/${EVAL_ID}

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{"camera1":{"type":"opencv","index_or_path":"/dev/video10","width":640,"height":360,"fps":260,"fourcc":"MJPG"},"camera2":{"type":"opencv","index_or_path":"/dev/video11","width":640,"height":360,"fps":260,"fourcc":"MJPG"},"camera3":{"type":"opencv","index_or_path":"/dev/video12","width":640,"height":360,"fps":260,"fourcc":"MJPG"}}' \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/${EVAL_ID} \
  --dataset.single_task="Pick the bottle (eval ${MODEL} ckpt${CKPT})" \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=1800 \
  --dataset.reset_time_s=10 \
  --dataset.video=true \
  --dataset.fps=60 \
  --dataset.push_to_hub=false \
  --policy.path=${POLICY} \
  --robot.disable_torque_on_disconnect=false
```

> **ACT** uses camera keys `front`/`side`/`wrist`. **SmolVLA** uses `camera1`/`camera2`/`camera3`. Must match training.

> `--policy.path` must be an **absolute path** (`$PWD/...`). Relative paths are interpreted as HuggingFace repo IDs.

---

## Data Collection Protocol

| Factor | What to vary | Importance |
|--------|-------------|------------|
| **Backgrounds** | Tablecloths, backdrops, floor coverings | #1 (dominant) |
| **Lighting** | Room lights on/off/dim + red LED always on | #2 |
| **Distractors** | Random objects near bottle, people walking through | #3 |
| **Null episodes** | Arm stationary, no bottle (5-10 episodes) | Prevents false positives |
| **Object pose** | Bottle angle, position, distance | Placement target stays fixed |

---

## Architecture Details

### Overlay Formula

```
Ĩ_t = (1 - α·M_t) ⊙ I_t + α·M_t ⊙ C

α = 0.45, C = [255, 255, 0] (yellow)
```

### Why ACT Benefits More Than SmolVLA

```
ACT path:     Ĩ_t → conv1 [64,3,7,7] (TRAINABLE, 9408 params)
                   → ResNet-18 layers (TRAINABLE, 11.18M)
                   → Transformer enc/dec (TRAINABLE)
                   → action

SmolVLA path: Ĩ_t → SigLIP patch_embed [768,3,16,16] (FROZEN)
                   → SigLIP ViT 12-layer (FROZEN, 85.1M)
                   → SmolLM2 16-layer (FROZEN, 251.6M)
                   → Action Expert 16-layer (TRAINABLE, 98.2M)
                   → action
```

ACT's trainable conv1 adapts directly to the overlay color signature. SmolVLA's frozen SigLIP compresses each 16x16 patch into a single token, losing fine-grained overlay detail before the signal reaches the trainable Action Expert.

---

## Known Limitations

**Single-task only.** All experiments use one task (pick bottle, place to side). Multi-task evaluation (e.g., BEHAVIOR benchmark) is not conducted. Extending SEVO to multi-task settings would require per-class overlay differentiation and broader task coverage in training data, neither of which is validated here.

**Single object category.** The YOLO overlay highlights all detected "bottle" instances with the same color. In scenes with multiple object types requiring different actions, the current overlay cannot convey which object to manipulate. A per-class color scheme or language-conditioned detection would be needed.

**No comparison with large-scale VLAs.** We evaluate on ACT (51.60M) and SmolVLA (450.05M) because they run on our edge hardware. We do not compare with π0.5, OpenVLA, GR00T, or other billion-parameter models that target general-purpose manipulation.

**Training distribution bounds robustness.** SEVO does not guarantee generalization to visual conditions absent from training data. Our extreme-environment results (10% on white floors when trained only on dark floors) demonstrate this explicitly.

**Static grasping only.** The mobile base must stop before the arm executes. Grasping while moving drops success to approximately 10%.

**YOLO dependency.** Performance depends on detection quality. Novel object categories or heavy occlusion degrade the overlay.

---

## Citation

```bibtex
@inproceedings{sevo2026iros,
  title={{SEVO}: Semantic-Enhanced Virtual Observation for Cross-Environment Robot Manipulation},
  author={Fang, Tianchonghui and Zhuang, Yuan and Miao, Fei},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2026}
}
```
