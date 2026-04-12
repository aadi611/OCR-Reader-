# Handwritten OCR — PaddleOCR-based System

A production-ready pipeline for recognising handwritten text using [PaddleOCR 3.x](https://github.com/PaddlePaddle/PaddleOCR). Targets near state-of-the-art accuracy across diverse handwriting styles (neat, cursive, messy) with support for fine-tuning on custom datasets.

---

## Features

- **Detection** — DB++ (Differentiable Binarization) with LKPAN neck, tuned for irregular handwriting baselines
- **Recognition** — SVTR-LCNet with CTC + SAR dual loss, `48×320` input for tall ascenders/descenders
- **Preprocessing** — Hough-based deskew, adaptive thresholding, morphological cleanup
- **Augmentation** — Elastic distortion, perspective warp, ink degradation simulation via Albumentations
- **Post-processing** — SymSpell spell correction, regex-based OCR error rules
- **Language model** — KenLM CTC beam search decoder + GPT-2 candidate reranking
- **API server** — FastAPI with async thread-pool inference
- **Evaluation** — CER, WER, sequence accuracy with bootstrap confidence intervals

---

## Project Structure

```
├── configs/
│   ├── det/handwriting_det.yml      # DB++ detection config
│   └── rec/handwriting_rec_svtr.yml # SVTR recognition config
├── dict/
│   └── en_dict.txt                  # Character vocabulary
├── src/
│   ├── preprocess.py                # Deskew + adaptive threshold pipeline
│   ├── augment.py                   # Handwriting augmentation + synthetic data gen
│   ├── evaluate.py                  # CER / WER / bootstrap CI metrics
│   ├── postprocess.py               # Spell correction + OCR error rules
│   ├── language_model.py            # KenLM + GPT-2 reranking
│   └── writer_adversarial.py        # Writer-adversarial training head
├── train_data/                      # Put your dataset here (see Dataset Setup)
├── infer.py                         # CLI inference
├── server.py                        # FastAPI inference server
├── train.py                         # Fine-tuning wrapper
├── export.py                        # Export to Paddle inference / ONNX
└── download_pretrained.py           # Download PP-OCRv4 pretrained weights
```

---

## Requirements

- Python 3.9–3.11
- CUDA 11.8 or 12.3 (for GPU) — CPU-only also supported
- ~4 GB disk for models and dependencies

---

## Installation

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 3. Install PaddlePaddle (GPU — CUDA 12.3)
pip install paddlepaddle-gpu==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/

# For CPU only:
# pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Download pretrained models
python download_pretrained.py
```

---

## Quick Start — Inference

```bash
python infer.py --image path/to/handwriting.jpg
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | required | Input image path |
| `--det-model` | PP-OCRv5 default | Custom detection model dir |
| `--rec-model` | PP-OCRv5 default | Custom recognition model dir |
| `--no-gpu` | GPU on | Force CPU inference |
| `--conf-threshold` | `0.5` | Discard predictions below this score |
| `--save-output` | — | Save annotated image to path |
| `--no-preprocess` | — | Skip deskew + threshold preprocessing |

---

## API Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

**POST** `/ocr` — multipart image upload, returns:

```json
{
  "text": "full recognised string",
  "words": [
    { "text": "hello", "confidence": 0.973, "bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] }
  ],
  "avg_confidence": 0.961
}
```

**GET** `/health` — liveness check.

---

## Fine-tuning on Your Own Data

### 1. Prepare dataset

Supported public datasets: [IAM](https://fki.inf.unibe.ch/databases/iam-handwriting-database), [CVL](https://cvl.tuwien.ac.at/research/cvl-databases/an-on-line-computer-generated-visual-text-recognition-database/), [RIMES](http://www.rimes-database.fr/).

For custom data, label with [Label Studio](https://labelstud.io/) using the OCR template and export in PaddleOCR format.

Place files as:
```
train_data/
  rec/
    train/          ← word-level image crops
    val/
    train_label.txt ← format: image_path\ttext
    val_label.txt
  det/
    train/          ← full document images
    val/
    train.json
    val.json
```

### 2. Train

```bash
# Both detection and recognition
python train.py --stage both

# Recognition only (most common for fine-tuning)
python train.py --stage rec

# Multi-GPU
python train.py --stage both --gpus 0,1,2,3
```

### 3. Export

```bash
# Export to Paddle static inference format
python export.py --model-dir output/rec_handwriting_svtr/ --format paddle

# Export to ONNX
python export.py --model-dir output/rec_handwriting_svtr/ --format onnx
```

---

## Evaluation

```python
from src.evaluate import evaluate_model, character_error_rate

# Quick CER on a list of predictions vs ground truths
cer = character_error_rate(predictions, ground_truths)
print(f"CER: {cer:.2%}")
```

**Benchmark targets:**

| Condition | CER target |
|-----------|-----------|
| Neat handwriting | < 5% |
| Typical handwriting | < 12% |
| Production with LM | < 5% |
| SOTA (IAM, with LM) | ~2.1% |

---

## Data Augmentation

```python
from src.augment import get_handwriting_augmentation

aug = get_handwriting_augmentation()
augmented = aug(image=img)["image"]
```

Includes: elastic distortion, perspective warp, ink noise, blur, brightness/contrast jitter, compression artefacts, coarse dropout.

---

## Accuracy Improvement Roadmap

```
Baseline PP-OCRv5 (no fine-tuning)  →  CER ~20–35% on handwriting
+ Fine-tune on IAM                  →  CER ~10–15%
+ Custom domain data                →  CER ~8–12%
+ SVTR-Large architecture           →  CER ~6–10%
+ Better augmentation               →  CER ~5–9%
+ KenLM language model              →  CER ~3–6%
+ Spell correction post-processing  →  CER ~2–5%
```

---

## Common Mistakes

- Using default `32px` image height — use `48px` for handwriting (ascenders/descenders)
- Evaluating on writers seen during training — always hold out full writers
- Applying spell correction unconditionally — only correct predictions with confidence < 0.7
- Skipping textline orientation correction — handwriting is often tilted

---

## License

MIT
