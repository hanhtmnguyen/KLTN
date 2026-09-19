# PI-GAN Data Assimilation Framework

**Supervised Physics-Informed GAN with Multi-Source Remote Sensing Data Assimilation
for Flood Inundation Mapping on the Red River, Hanoi**

## Overview

This framework implements a **Supervised Physics-Informed Generative Adversarial Network
(PI-GAN)** that functions as a **Hybrid Surrogate Model** for 2D flood simulation. The
model learns a direct mapping from low-fidelity HAND flood models to high-fidelity
HEC-RAS 2D simulations, regularised by:

- **2D Shallow Water Equations (SWE)** — physics-informed regularization
- **GEOGloWS discharge** — upstream boundary condition data assimilation
- **ICESat-2 ATL13 WSE** — downstream boundary condition data assimilation
- **Sentinel-1 SAR** — spatial flood extent data assimilation

## Architecture

```
L_total = λ_pixel   · L_pixel       ← Supervised by HEC-RAS (PRIMARY)
        + λ_physics  · L_physics     ← Unsteady SWE regularization
        + λ_upstream · L_upstream    ← GEOGloWS Q(t) BC constraint
        + λ_downstream · L_downstream ← ICESat-2 ATL13 WSE constraint
        + λ_adv      · L_adv         ← GAN adversarial loss
        + λ_SAR      · L_SAR         ← Sentinel-1 flood extent
```

## Project Structure

```
da-framework/
├── configs/              # Pipeline configuration
├── src/
│   ├── utils/            # Geospatial utilities, logging, visualization
│   ├── data_acquisition/ # FABDEM, ICESat-2 (ATL03+ATL13), GEOGloWS, Sentinel
│   ├── preprocessing/    # DTM correction, terrain features, roughness, HAND
│   ├── groundtruth/      # HEC-RAS automation + training data builder
│   ├── model/            # PI-GAN: generator, discriminator, losses, trainer
│   └── assessment/       # Metrics, scenario runner, risk mapping
├── scripts/              # Pipeline execution scripts
├── data/                 # Raw + processed data (gitignored)
├── models/               # Trained model checkpoints
└── outputs/              # Results, risk maps, figures
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python -m src.model.trainer --config configs/base_config.yaml
```

## Training Curriculum

| Phase | Epochs | Active Losses |
|-------|--------|---------------|
| **A** (warm-up) | 1–50 | L_pixel only |
| **B** (physics + BC) | 51–150 | L_pixel + L_physics + L_upstream + L_downstream |
| **C** (full DA-GAN) | 151+ | All losses active |

## Key Innovation

The framework assimilates multi-source remote sensing data into the supervised
training loop as **regularization constraints**, enabling the model to produce
physically consistent flood maps that are anchored to real-world observations
from global hydrological models and satellite altimetry.
