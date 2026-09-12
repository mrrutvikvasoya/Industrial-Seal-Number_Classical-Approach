# Industrial Seal Number OCR — Classical Approach

Reads the **7-digit code** stamped on an industrial security seal from a 1920×1200
grayscale photo, using **only classical computer vision and classical machine
learning** — no neural network, no GPU, no cloud. Built for the BTHA "AI and Industry"
2026 Summer School competition.

> **Test accuracy: 1,412 / 1,414 = 99.86% exact match** (whole 7-digit code correct),
> 99.98% per-digit, 0 localization misses, ~80 ms/image on CPU.

---

## Overview

Given a photo of a metal seal — with glare, tilt, surrounding hardware, and a `TESCO`
distractor stamp — the pipeline finds the row of seven digits, cleans each one, and
classifies it with a Support Vector Machine. It is the **classical** solution
(Chapter 2 of the task) among the three suggested approaches (Classical / CNN / VLM).

Why classical:
- **Runs anywhere** — CPU-only, offline, no CUDA, no large dependencies.
- **Fast** — ~80 ms/image, well under a one-part-per-second production line.
- **Interpretable** — every stage is inspectable and explainable.

## Results

| Split | Exact match | Per-digit | Localizer misses |
|------:|:-----------:|:---------:|:----------------:|
| Validation (1,414) | 1,408 / 1,414 = **0.9958** | 0.9994 | 0 |
| **Test (1,414)** | **1,412 / 1,414 = 0.9986** | **0.9998** | **0** |

- Localization finds all seven digits on **100%** of train/val/test seals.
- The two test errors are single-digit slips on physically ambiguous stamps (e.g. a
  smudged `8` read as `0`), not pipeline bugs.
- The test split is a held-out set the model never saw during training or tuning.

## How it works

```
PNG ─▶ read_gray ─▶ find_digits ─▶ crop_digit ×7 ─▶ normalize_digit ×7
    ─▶ features ×7 ─▶ StandardScaler ─▶ SVC(rbf) ─▶ 7 digits ─▶ CSV
```

1. **Decode** — load the PNG straight to grayscale (`dataio.read_gray`).
2. **Localize** (`localize.find_digits`) — Otsu / adaptive thresholding → morphology →
   connected components → keep digit-shaped blobs → group seven that form a straight,
   evenly-spaced row. Rejects the `TESCO` text and hardware by geometry, with a
   fallback ladder for tilt and broken strokes. Returns 7 boxes + row angle.
3. **Crop & normalize** (`crops.py`) — cut each digit (with margin, deskew), then
   normalize to a **32×32, bright-ink-on-black** canvas. The *same* function is used
   for training and inference, which removes the domain gap.
4. **Features** (`features.py`) — turn each crop into a fixed **343-dim vector**:
   HOG (324) + zoning 4×4 (16) + hole topology (2) + aspect (1).
5. **Recognize** (`recognize.py`) — one batched `StandardScaler → RBF-SVM` call reads
   all seven digits, joined into the code.
6. **Write** (`main.py`) — process each image **one at a time** (production-line style),
   using all CPU cores per image, and write `<team>.csv`.

The model is trained on **~69,000 digit crops harvested** from the training seals: each
seal is localized and its seven left-to-right boxes are labeled with the seven digits of
its known code. Training uses 25,000 balanced crops (2,500/class) plus one augmented
copy each; augmentation (rotation, scale, blur, shadow, gamma) is applied to training
crops only. Train, validation, and test crops come from disjoint seal splits — no
leakage.

## Project structure

```
mywork/
├── main.py            Competition entry point: folder of PNGs → <team>.csv
├── harvest.py         Build labeled digit crops from a localized split
├── train.py           Train the SVM from harvested crops
├── eval.py            Score end to end (exact match, per-digit, attribution)
├── requirements.txt   Pinned dependencies
├── run.md             Copy-paste commands
├── weights/
│   └── svm.joblib     The trained model (ships with the repo)
└── seals/             Library package
    ├── dataio.py        read images + label manifests
    ├── thresholds.py    binary masks (Otsu / adaptive)
    ├── rows.py          group tilted components into a digit row
    ├── localize.py      find the 7 digit boxes
    ├── crops.py         cut + normalize one digit (shared by train & inference)
    ├── augment.py       training-only crop augmentation
    ├── features.py      HOG + zoning + holes + aspect → 343-dim vector
    ├── recognize.py     localize → classify → 7-digit string
    ├── dataset.py       load harvested crops into a feature matrix
    └── config.py        constants
```

## Getting started

Requires **Python 3.12**. Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `opencv-python`, `scikit-learn`, `scipy`, `joblib`,
`threadpoolctl` (all pinned; CPU-only).

### Run on a folder of seal PNGs (the competition interface)

```bash
python main.py --input-dir <folder-of-pngs> --output-dir <out> --team <name>
```

Writes `<out>/<name>.csv` with header `filename;number`. Every image gets a non-empty
result; unreadable images fall back to a guess instead of aborting the run.

### Evaluate against ground truth

```bash
python eval.py --input-dir <folder> --labels <gt.csv>
```

### Reproduce the model

```bash
python harvest.py --input-dir <dataset>/train --labels <dataset>/splits/split_seals/train.csv --output-dir outputs/harvest_train
python harvest.py --input-dir <dataset>/val   --labels <dataset>/splits/split_seals/val.csv   --output-dir outputs/harvest_val
python train.py --train outputs/harvest_train/harvest.csv --val outputs/harvest_val/harvest.csv
```

See [`run.md`](run.md) for the exact commands used on the development machine.

## The technique (classical recipe)

| Stage | Method |
|---|---|
| Thresholding | Otsu (+ adaptive Gaussian fallback) |
| Segmentation | connected components + morphology + row alignment |
| Normalization | 32×32, aspect-preserved, consistent polarity |
| Features | HOG (9 bins, 8×8 cells) + zoning 4×4 + hole topology + aspect |
| Classifier | RBF Support Vector Machine |

## Notes and limitations

- Accuracy figures are on the local val/test splits; a more degraded hidden test may
  score lower. Under strong synthetic degradation (blur, rotation, noise, shadow) the
  model still held ~99.5% exact match.
- The remaining errors are physically ambiguous digits — the last fraction of a percent
  is a data limit, not a code bug.
- Per-crop training labels are *derived* from the seal code and box order, so a
  mis-localized box yields a mislabeled crop (a rare "noisy harvest" the SVM tolerates).

## License

Provided for the BTHA Summer School 2026 competition.
