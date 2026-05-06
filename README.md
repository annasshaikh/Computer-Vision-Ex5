# Assignment 5 — CNNs & Neural Style Transfer

## Environment

```bash
conda create -n a5 python=3.10
conda activate a5
pip install -r requirements.txt
```

Or with conda:
```bash
conda env create -f environment.yml
conda activate a5
```

Hardware used: NVIDIA GPU recommended (CUDA 11.8+). CPU fallback works but is slow.

---

## Directory structure

```
assignment5/
├── task1_cnn/
│   ├── config.yaml          # all hyperparameters + seed
│   ├── data.py              # dataset loader + augmentation
│   ├── models.py            # Model A (BaselineCNN) + Model B (DeepRegCNN)
│   ├── train.py             # training loop (Adam vs SGD, early stopping)
│   ├── evaluate.py          # metrics, plots, comparison table
│   ├── weights/             # saved .pth files (after training)
│   ├── logs/                # TensorBoard + CSV logs
│   └── cnn_outputs/         # confusion matrices, curves, comparison CSV
│
├── task2_nst_video/
│   ├── config.yaml
│   ├── nst.py               # Gatys NST, β/α sweep, layer ablation, grid
│   ├── video_pipeline.py    # matting + NST + compositing → 3 output videos
│   ├── matting/
│   │   ├── model.py         # MobileNetV2-UNet alpha matte predictor
│   │   └── train.py         # matting training on AISegment
│   ├── content/             # 5 extracted frames (add after recording)
│   ├── style/               # 3 style paintings + README.md
│   │   └── README.md        # sources & licenses
│   ├── input_video.mp4      # YOUR raw recording (add manually)
│   └── outputs/             # all generated images and videos
│
├── assignment5_notebook.ipynb
├── requirements.txt
└── README.md
```

---

## Task 1 — CNN on Seeds Dataset

### 1. Place your dataset
```
seeds/
    1.jpg
    2.jpg
    ...
    150.jpg
```

### 2. Train (Adam vs SGD comparison + Model B)
```bash
cd task1_cnn
python train.py --all
```

Or individually:
```bash
python train.py --model a --opt adam
python train.py --model a --opt sgd
python train.py --model b
```

### 3. Evaluate and generate comparison table
```bash
python evaluate.py --compare
```

---

## Task 2 — NST + Matting + Video

### A. Download style images
See `task2_nst_video/style/README.md` for download commands.

### B. Record your video
Place it at `task2_nst_video/input_video.mp4`.

### C. Extract content frames
```bash
python task2_nst_video/video_pipeline.py --extract_frames --video task2_nst_video/input_video.mp4
```

### D. Train the matting model (requires AISegment dataset)
```bash
python task2_nst_video/matting/train.py \
    --data   data/aisegment \
    --epochs 30 \
    --out    task2_nst_video/matting/weights
```

### E. NST sanity check & ablations
```bash
# Single transfer
python task2_nst_video/nst.py \
    --content task2_nst_video/content/frame_01.jpg \
    --style   task2_nst_video/style/starry_night.jpg

# β/α sweep
python task2_nst_video/nst.py --sweep_ratios \
    --content task2_nst_video/content/frame_01.jpg \
    --style   task2_nst_video/style/starry_night.jpg \
    --out_dir task2_nst_video/outputs

# Layer ablation
python task2_nst_video/nst.py --layer_ablation \
    --content task2_nst_video/content/frame_01.jpg \
    --style   task2_nst_video/style/starry_night.jpg \
    --out_dir task2_nst_video/outputs

# 5x3 grid
python task2_nst_video/nst.py --grid \
    --out_dir task2_nst_video/outputs

# Feature maps
python task2_nst_video/nst.py --feature_maps \
    --image   task2_nst_video/content/frame_01.jpg \
    --out_dir task2_nst_video/outputs
```

### F. Full video pipeline
```bash
python task2_nst_video/video_pipeline.py \
    --video   task2_nst_video/input_video.mp4 \
    --style   task2_nst_video/style/starry_night.jpg \
    --matting task2_nst_video/matting/weights/matting_best.pth \
    --out_dir task2_nst_video/outputs \
    --nst_steps 300 \
    --beta 1e5
```

Outputs:
- `outputs/stylized_background.mp4`  — Variant 1: background stylized, subject natural
- `outputs/stylized_subject.mp4`     — Variant 2: subject stylized, background natural
- `outputs/stylized_full.mp4`        — Variant 3: whole frame stylized
- `outputs/branded_poster.png`       — 1024×1024 marketing still

---

## Notebook

Open `assignment5_notebook.ipynb` for an end-to-end walkthrough of both tasks including all plots and analysis.
