# Columbia Research - Diffusion Posterior Sampling (DPS) Implementation

This repository contains an implementation of Diffusion Posterior Sampling (DPS) for solving inverse problems using pre-trained diffusion models.

## Overview

DPS leverages pre-trained diffusion models to solve various inverse problems without requiring task-specific training. This implementation is based on OpenAI's guided-diffusion codebase and includes modifications for coherent imaging applications.

## File Structure

```
dps/
├── configs/
│   ├── dct_speckle_config.yaml
    ...
│   └── inpainting_config.yaml        
├── guided_diffusion/
│   ├── condition_methods.py
│   └── measurements.py
├── models/           # Pre-trained model checkpoints
│   └── ffhq_10m.pt
├── results/              # Output directory
├── sample_condition.py
└── sample_condition.sh
```

## Quick Start

### 1. Download Pre-trained Models

**FFHQ Checkpoint**: Download the pre-trained $256\times256$ FFHQ diffusion model checkpoint from:
- [OpenAI's guided-diffusion releases](https://github.com/openai/guided-diffusion)
- Direct link:  [`ffhq_10m.pt`](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh)

Place the checkpoint in the `models/` directory:
```
dps/models/
└── ffhq_10m.pt
```


