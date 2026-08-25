# This file consists of how to teleoperate and autonomously harvest strawberry through pedicel-targeted grasping 
# Pedicel-Targeted Strawberry Harvesting via Behavioral Cloning

<p align="center">
  <img src="assets/teaser.png" alt="Pedicel-Targeted Strawberry Harvesting" width="800"/>
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/nitindominicrai/strawberry_harvest_v5_20260814_150933">
    <img src="https://img.shields.io/badge/🤗%20Dataset-HuggingFace-orange" alt="Dataset"/>
  </a>
  <a href="https://huggingface.co/nitindominicrai">
    <img src="https://img.shields.io/badge/🤗%20Models-HuggingFace-blue" alt="Models"/>
  </a>
  <a href="https://arxiv.org/abs/XXXX.XXXXX">
    <img src="https://img.shields.io/badge/Paper-IROS%202026-green" alt="Paper"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.12-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>
</p>

> **Paper:** *Pedicel-Targeted Strawberry Harvesting via Imitation Learning on a Low-Cost Teleoperation Platform*  
> Nitin Rai, Won Suk Lee — University of Florida  
> IROS Agribotics Workshop 2026, Pittsburgh, Pennsylvania

---

## Overview

This repository contains the full pipeline for autonomous strawberry pedicel harvesting using behavioral cloning on a low-cost robotic arm. The system learns to grasp the strawberry pedicel (1.4–2.4 mm diameter) from human teleoperation demonstrations, bypassing the need for explicit computer vision detection or motion planning.

**Key features:**
- End-to-end visuomotor policy from wrist-camera input to joint-space actions
- ACT (Action Chunking with Transformers) policy with ResNet-50 visual backbone
- Trained on 85+ human teleoperation demonstrations
- Deployed autonomously on Jetson Orin AGX at 15 Hz — no cloud compute
- Open-access dataset on HuggingFace Hub

---

## Hardware Requirements

| Component | Specification |
|---|---|
| Robot arm | HiWonder SO-ARM101 (leader + follower) |
| Edge compute | NVIDIA Jetson Orin AGX (64 GB) |
| Camera | USB wrist-mounted camera (640×480, 30 fps) |
| Training GPU | NVIDIA B200 (HiPerGator HPC) |
| OS | Ubuntu 20.04, JetPack 5.x |
| CUDA | 11.4 (Jetson custom build) |

---

## Software Requirements

```
Python        3.12
PyTorch       2.1.0 (built from source for CUDA 11.4 on aarch64)
LeRobot       HiWonder fork
torchvision   custom build (no C++ extensions)
```

---

## Repository Structure

```
autonomous_strawberry_pedicel_harvesting/
├── README.md
├── INSTALL.md                   ← step-by-step setup guide
├── scripts/
│   ├── record_wrist_cam.py      ← standalone wrist camera recorder
│   ├── convert_to_fp16.py       ← convert trained model to FP16
│   └── benchmark_inference.py   ← measure inference speed on Jetson
├── analysis/
│   └── generate_all_figures.py  ← reproduce all paper figures
├── assets/
│   ├── teaser.png
│   └── pipeline.png
└── configs/
    └── act_config.json          ← training hyperparameters
```

---

## Dataset

The demonstration dataset is publicly available on HuggingFace:

| Version | Episodes | Frames | Resolution | Notes |
|---|---|---|---|---|
| [v5](https://huggingface.co/datasets/nitindominicrai/strawberry_harvest_v5_20260814_150933) | 31 | 38,320 | 640×480 | Clean baseline |
| [v7](https://huggingface.co/datasets/nitindominicrai/strawberry_harvest_v7) | 131 | 227,108 | 480×640 | Extended dataset |

Each episode contains synchronized:
- Wrist camera frames (MP4, 30 fps)
- 6-DOF joint positions (action + observation)
- Timestamps and episode metadata

---

## Trained Models

Pre-trained model weights are hosted on HuggingFace:

| Model | Backbone | Steps | Dataset | Download |
|---|---|---|---|---|
| ACT-ResNet50 | ResNet-50 | 100K | v5 (31 eps) | [🤗 Link](https://huggingface.co/nitindominicrai) |
| ACT-ResNet50-FP16 | ResNet-50 (FP16) | 100K | v5 (31 eps) | [🤗 Link](https://huggingface.co/nitindominicrai) |

---

## Installation

See [INSTALL.md](INSTALL.md) for the full step-by-step setup guide.

**Quick start (assumes LeRobot already installed):**

```bash
# Clone this repository
git clone https://github.com/nitin-dominic/autonomous_strawberry_pedicel_harvesting
cd autonomous_strawberry_pedicel_harvesting

# Download trained model from HuggingFace
hf download nitindominicrai/act_strawberry_model \
    --local-dir models/pretrained_model \
    --repo-type model
```

---

## Usage

### 1. Teleoperation (Data Collection)

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ handeye: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, rotation: 90}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --dataset.repo_id=YOUR_HF_USERNAME/strawberry_harvest \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick the strawberry pedicel" \
    --dataset.reset_time_s=60 \
    --dataset.push_to_hub=true
```

### 2. Training (HiPerGator or any GPU)

```bash
python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=YOUR_HF_USERNAME/strawberry_harvest \
    --policy.type=act \
    --output_dir=outputs/act_strawberry \
    --policy.device=cuda \
    --steps=100000 \
    --batch_size=32 \
    --dataset.video_backend=pyav \
    --dataset.use_imagenet_stats=false \
    --policy.vision_backbone=resnet50 \
    --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V2
```

### 3. Autonomous Inference (Jetson Orin)

```bash
lerobot-rollout \
    --strategy.type=base \
    --policy.path=models/pretrained_model \
    --policy.device=cuda \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ handeye: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, rotation: 90}}" \
    --task="Pick the strawberry pedicel" \
    --duration=45
```

---

## Results

| Condition | Trials | Success Rate |
|---|---|---|
| Fixed position | 10 | — |
| Varied (±3 cm) | 10 | — |
| With occlusion | 10 | — |
| Full pot | 10 | — |

*Results to be updated after physical trial completion.*

**Deployment characteristics:**
- Inference rate: 15–16 Hz (FP32) / 19–20 Hz (FP16)
- Platform: Jetson Orin AGX, CUDA 11.4
- No cloud compute dependency

---

## Reproducing Paper Figures

```bash
# Install dependencies
pip install scikit-learn matplotlib pandas numpy

# Generate all figures
python analysis/generate_all_figures.py
```

Figures are saved to `outputs/figures/` as both PDF (for LaTeX) and PNG.

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{rai2026pedicel,
  title     = {Pedicel-Targeted Strawberry Harvesting via Imitation Learning
               on a Low-Cost Teleoperation Platform},
  author    = {Rai, Nitin and Lee, Won Suk},
  booktitle = {Proc. IEEE/RSJ IROS Workshop on Agricultural Robotics (Agribotics)},
  year      = {2026},
  address   = {Pittsburgh, Pennsylvania}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [LeRobot](https://github.com/huggingface/lerobot) by HuggingFace
- [HiWonder SO-ARM101](https://www.hiwonder.com)
- University of Florida Agricultural and Biological Engineering
- UF Research Computing (HiPerGator)
