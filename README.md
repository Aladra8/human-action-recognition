# Human Action Recognition for Human-Robot Interaction
### Classical Computer Vision Pipeline — KTH Action Dataset

> **Course Project** · University of Padua · Department of Information Engineering (DEI)  
> **Student (Author):** Baba Drammeh `2085440`

---

## Overview

A classical computer vision system that classifies six human actions from video sequences — **walking, jogging, running, boxing, hand waving, hand clapping** — using only traditional CV techniques (no deep learning). Designed for Human-Robot Interaction (HRI) applications where real-time, interpretable motion understanding is critical.

The pipeline processes 40-frame sequences from the KTH Action Recognition Dataset and outputs:
- A **bounding box** tracking the human actor across all 40 frames
- A **single global action label** for the entire sequence
- Performance metrics: mIoU, accuracy, per-class F1, confusion matrix

---

## Results

| Metric | Value |
|--------|-------|
| **Global Accuracy (LOOCV)** | **~62%** |
| **mIoU (frame 20)** | **~0.73** |
| Macro F1 | ~0.60 |
| Walking F1 | ~0.83 |
| Dataset sequences | 72 (12 per class × 6 classes) |
| Evaluation method | Leave-One-Out Cross Validation |

---

## Pipeline

```
Raw AVI frames
      │
      ▼
┌─────────────────────────────────┐
│  1. Smart Frame Selection       │  Scored window search across the full
│     (download_dataset.py)       │  video — picks 40 frames with most
│                                 │  stable, visible human foreground.
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  2. Background Subtraction      │  Pixel-wise median model over all
│     (BackgroundModel)           │  40 frames → binary foreground mask.
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  3. Human Detection & Tracking  │  Largest-blob detection + exponential
│     (HumanDetector)             │  moving-average bbox smoothing.
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  4. Feature Extraction (36-dim) │  Velocity · Optical flow · Shape ·
│     (FeatureExtractor)          │  FFT stride frequency · Discriminative
│                                 │  extras (asymmetry, upward flow, …)
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  5. RBF-SVM Classification      │  Min-max normalised features →
│     (ActionClassifier)          │  OpenCV SVM with LOOCV evaluation.
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  6. Evaluation & Visualisation  │  mIoU · Accuracy · F1 · Confusion
│     (Evaluator)                 │  matrix · Annotated output images.
└─────────────────────────────────┘
```

---

## Feature Vector (36 dimensions)

| Group | Features | Discriminates |
|-------|----------|--------------|
| **f0–f7** Centroid velocity | Mean/std/max speed, total displacement, h/v ratio | Locomotion vs stationary |
| **f8–f19** Optical flow stats | Magnitude, direction, upper/lower body zones, convergence | Clapping vs waving, running vs walking |
| **f20–f24** Bbox shape | Aspect ratio, area, height (normalised) | Pose-based |
| **f25–f28** FFT frequency | Dominant stride frequency from height + centroid-y oscillation | Jogging vs running |
| **f29–f35** Discriminative | Asymmetry, upward flow, flow variability, acceleration, scale-normalised speed, vertical flow centroid | Boxing vs waving vs clapping; jogging vs running (scale-free) |

---

## Project Structure

```
Cvision/
├── include/                    ← C++ header files
│   ├── types.hpp               — shared structs & action class IDs
│   ├── dataset_loader.hpp      — loads 40-frame sequences from disk
│   ├── detector.hpp            — background model + human tracker
│   ├── feature_extractor.hpp   — 36-dim classical CV feature vector
│   ├── action_classifier.hpp   — RBF-SVM with LOOCV
│   └── evaluator.hpp           — IoU, accuracy, F1, confusion matrix
├── src/                        ← C++ implementation files
│   ├── main.cpp
│   ├── dataset_loader.cpp
│   ├── detector.cpp
│   ├── feature_extractor.cpp
│   ├── action_classifier.cpp
│   └── evaluator.cpp
├── CMakeLists.txt              ← CMake build configuration
├── download_dataset.py         ← Dataset processor (smart frame selection)
├── dataset_check.sh            ← Dataset structure validator
├── setup_macos.sh              ← One-shot macOS build script
├── setup_linux.sh              ← One-shot Linux build script
├── kth_raw/                    ← Raw KTH AVI files (user-provided)
├── dataset/                    ← Processed 72-sequence dataset (generated)
└── output_frames/              ← Annotated result images (generated)
```

---

## Requirements

| Dependency | Version | Install |
|-----------|---------|---------|
| C++ compiler | C++17 | `xcode-select --install` (macOS) |
| CMake | ≥ 3.16 | `brew install cmake` |
| OpenCV | ≥ 4.8 | `brew install opencv` |
| Python 3 + opencv-python | any | `pip3 install opencv-python` |

---

## Setup & Build (macOS)

```bash
# 1. Install all dependencies and build in one command
chmod +x setup_macos.sh && ./setup_macos.sh

# 2. Prepare dataset (after placing KTH AVI folders inside kth_raw/)
python3 download_dataset.py --raw ./kth_raw --output ./dataset

# 3. Validate dataset structure
chmod +x dataset_check.sh && ./dataset_check.sh ./dataset
```

---

## Running on mac using VsCode

```bash
# Full LOOCV evaluation + annotated output images (recommended)
./har ./dataset --loocv --visualize

# Train and save the SVM model, then load it next time (faster)
./har ./dataset --loocv --save-model model.svm
./har ./dataset --load-model model.svm --visualize

# Custom output paths
./har ./dataset --loocv \
      --visualize --output my_results/ --report my_report.csv

# Help
./har --help
```

---

## Dataset

The [KTH Action Recognition Dataset](https://www.csc.kth.se/cvap/actions/) contains 600 video files across 25 subjects, 6 action classes, and 4 environmental scenarios. This project uses a stratified subset of **72 sequences** (12 per class, 3 from each scenario d1–d4).

The `download_dataset.py` script processes the raw AVI files: it scores every possible 40-frame window in each video and selects the window where the human actor is most consistently and stably visible, avoiding setup/transition frames.

---

## Output

Each annotated image in `output_frames/` shows:
- **Red bounding box** — system's predicted human localisation at frame 20
- **Label (top-right)** — predicted action class name

The `eval_report.csv` contains all numeric metrics for further analysis.

---

## Ubuntu Setup

```bash
sudo apt install -y build-essential cmake libopencv-dev
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) && cd ..
python3 download_dataset.py --raw ./kth_raw --output ./dataset
./har ./dataset --loocv --visualize
```
