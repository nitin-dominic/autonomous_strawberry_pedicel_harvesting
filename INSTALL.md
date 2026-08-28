# Installation Guide

## Requirements

- HiWonder SO-ARM101 (leader + follower arms)
- NVIDIA Jetson Orin AGX with JetPack 5.x
- USB wrist-mounted camera
- Miniconda installed on Jetson
- HuggingFace account (free)

---

## Step 1 — Clone Repository

```bash
git clone https://github.com/nitin-dominic/autonomous_strawberry_pedicel_harvesting
cd autonomous_strawberry_pedicel_harvesting
```

---

## Step 2 — Create Conda Environment

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
```

---

## Step 3 — Install PyTorch for Jetson Orin

Jetson Orin requires PyTorch built for aarch64 with CUDA 11.4.
Download and install the pre-built wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Verify:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Expected: CUDA: True
```

---

## Step 4 — Install LeRobot

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot && pip install -e ".[so101]" && cd ..
```

---

## Step 5 — Install Additional Dependencies

```bash
pip install huggingface_hub safetensors opencv-python pandas numpy
```

---

## Step 6 — Download Pre-trained Model

```bash
# Login to HuggingFace
hf auth login

# Download model weights
hf download nitindominicrai/act_strawberry_pedicel \
    --local-dir models/pretrained_model \
    --repo-type model

# Verify
ls models/pretrained_model/
# Expected: model.safetensors, config.json, train_config.json, ...
```

---

## Step 7 — Connect Hardware

```bash
# Set permissions for arm and camera
sudo chmod 666 /dev/ttyACM0   # follower arm
sudo chmod 666 /dev/ttyACM1   # leader arm

# Verify camera
ls /dev/video*                 # typically /dev/video0
```

---

## Step 8 — Calibrate Arms

```bash
# Calibrate follower
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm

# Calibrate leader
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm
```

During calibration — open the gripper fully and move each joint through its complete range when prompted.

---

## Step 9 — Test Teleoperation

```bash
conda activate lerobot

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ handeye: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, rotation: 180}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true
```

The follower arm should mirror the leader. Camera feed opens in Rerun viewer.

---

## Step 10 — Run Autonomous Inference

```bash
lerobot-rollout \
    --strategy.type=base \
    --policy.path=models/pretrained_model \
    --policy.device=cuda \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ handeye: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, rotation: 180}}" \
    --task="Pick the strawberry pedicel" \
    --duration=45
```

---

## Troubleshooting

**Camera rejects resolution:**
```bash
python -c "
import cv2
cap = cv2.VideoCapture(0)
print(int(cap.get(3)), int(cap.get(4)))
cap.release()
"
# Use the printed width x height in --robot.cameras config
```

**Inference too slow (<10 Hz):**
```bash
python scripts/convert_to_fp16.py
# Use the FP16 model path in rollout
```

**Motor overload on disconnect:**
Reduce gripper torque limit before running rollout:
```bash
python -c "
import sys; sys.path.insert(0, 'lerobot/src')
from lerobot.motors.feetech.feetech import FeetechMotorsBus
bus = FeetechMotorsBus(port='/dev/ttyACM0',
    motors={'gripper': (6, 'sts3215')})
bus.connect()
bus.write('Torque_Limit', 'gripper', 400)
bus.disconnect()
print('Done')
"
```

**FrameTimestampError during training:**
Add `--dataset.tolerance_s=100.0` to your training command.

---

## Camera Configuration Reference

| Dataset | width | height | rotation | Tensor shape |
|---|---|---|---|---|
| v7 (this work) | 640 | 480 | 180° | [480, 640, 3] |

Always use identical camera config for recording and inference.
