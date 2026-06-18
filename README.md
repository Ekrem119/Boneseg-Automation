# BoneSeg: Automated & Manual Computational Pipeline for Forensic Bio-Imaging

---

## Project Overview
`BoneSeg` is a Python-based software application designed for high-throughput forensic histomorphometry. By integrating Meta’s state-of-the-art **Segment Anything Model 2 (SAM 2)**, the pipeline flattens 3D confocal microscope Z-stack (.nd2) files into 2D Maximum Intensity Projections (MIP) and extracts quantitative metrics to analyze post-mortem tissue degradation (**bone diagenesis**).

The core diagnostic engine computes the **Inside-Object / Inside-Border (IO/IB)** fluorescence intensity ratio across multiple laser channels (DAPI/405nm, EGFP/488nm, Cy3/561nm) to assist in discriminating modern forensic remains (**ANAT**) from old archaeological samples (**KOEK**).

---

## Technical Features
- **Dual Operating Framework Matrices:**
  - **Automated Batch Pipeline:** Recursively traverses complex laboratory directories, dynamically assigning groups based on path topology and tracking metrics automatically without human intervention.
  - **Guided Manual Validation Workspace:** Supports real-time, interactive, multi-click iterative feedback loops (Left-Click to add structures, Right-Click to extract noise) allowing expert-level oversight.
- **Microstructural Boundary Local Subtraction:** Automatically executes binary matrix dilations to construct tight local perimeters around targeted osteocyte lacunae or diagenetic alterations, making comparisons independent of environmental background noise.
- **Multidimensional Data Structuring:** Generates a clean structured spreadsheet containing quantitative observation arrays (Area, Circularity, and Multi-channel Intensity Profiles) optimized for downstream Principal Component Analysis (PCA) sorting.

---

## Technical Environment & Dependencies

### Core System Environment
- **Operating Language:** Python 3.10
- **Hardware Target:** CUDA Accelerated Execution (GPU optimization via `torch 2.1.2+cu118`)

### Primary Deployment Framework
- **Deep Learning Segmenter:** SAM 2 v1.0 (Local deployment architecture using fine-tuned weights)
- **Image Processing & Morphometry:** `scikit-image (0.21)`, `opencv-python (4.8)`, `nd2reader / pims (3.3.0)`
- **Configuration Management:** `hydra-core (1.3)` & `iopath / fvcore (0.1)`
- **Data Engineering & Analytics:** `pandas (2.1)`, `scikit-learn (1.3)`, `matplotlib (3.7)`
- **Graphical User Interface Layer:** `customtkinter (5.2)` & `Pillow (10.0)`
- **System Resource Infrastructure:** `psutil (5.9)` & `threadpoolctl (3.2)`

### Model Architecture Fine-Tuning Environment
For users re-initializing the fine-tuning pipelines or training the SAM 2 deep learning layers, the environment includes:
- `hydra-core 1.3.2`
- `fvcore 0.1.5` / `iopath 0.1.10`
- `torchvision 0.16`
- `tqdm 4.66`
- `pycocotools 2.0`

---

## Installation & Setup Strategy

### Step 1: Repository Layout Setup
Ensure your local workspace replicates the following architecture. Meta's SAM 2 configuration engine files and the fine-tuned `.pt` weight checkpoints must be aligned precisely within the `sam2` subdirectory:

```text
├── BoneSeg.py
├── requirements.txt
├── README.md
└── sam2/
    ├── sam2/
    │   └── configs/
    └── checkpoints/
        └── finetuned_checkpoints/
            ├── checkpoint_blanc_DAPI.pt
            └── checkpoint_objet_DAPI.pt
```
### 2. Dependency Ingestion
Initialize a virtual environment (Python 3.10) and install the requirement matrix directly:

# Bash
```pip install -r requirements.txt```

### 3. Execution
Launch the primary GUI application from your command terminal:

# Bash
```python BoneSeg.py```

To test the app, download the sample of data here: https://doi.org/10.5281/zenodo.18236099
