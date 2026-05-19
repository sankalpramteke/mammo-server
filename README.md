# DISHA — mammo-server

> **Local Hospital Training Node — Privacy-First Federated Learning**
>
> mammo-server is a FastAPI Python server that runs inside each hospital. Doctors upload mammogram images here, the AI model trains on them entirely in memory, and only a mathematically-noised summary of what it learned is sent to the global coordinator — never the images themselves.

---

## What Does This Do?

mammo-server is the **private training environment** that runs on a hospital's own machine. It:

1. Receives mammogram images from the hospital portal
2. Trains the AI model on those images (in RAM — nothing saved to disk)
3. Adds privacy-preserving noise to the model's learning (Differential Privacy)
4. Sends only the noised weight update to mammo-global
5. Deletes all image data immediately after training

Patient images **never leave the hospital**. This is the fundamental privacy guarantee.

---

## System Flow

```
Doctor uploads ZIP / images
         │
         ▼
┌─────────────────────────────────┐
│   POST /train-dataset           │
│                                 │
│  1. Extract images from ZIP     │
│     (in RAM — never on disk)    │
│                                 │
│  2. For each image:             │
│     • Apply CLAHE preprocessing │
│     • Train model (1 step)      │
│     • Delete image from memory  │
│     • Run garbage collector     │
│                                 │
│  3. Compute weight delta        │
│     (new_weights - old_weights) │
│                                 │
│  4. Apply Differential Privacy  │
│     • Clip L2-norm of delta     │
│     • Add Gaussian noise        │
│                                 │
│  5. SHA-256 hash the result     │
│                                 │
│  6. Send noised delta to        │
│     mammo-global aggregator     │
│                                 │
│  7. Return audit proof to UI    │
└─────────────────────────────────┘
         │
         ▼
Images permanently gone from memory
Only the weight hash remains
```

---

## Technology Stack

| Technology | Purpose |
|---|---|
| **FastAPI** | High-performance Python web framework |
| **TensorFlow / Keras** | ResNet50-based mammogram classification model |
| **Pillow (PIL)** | Image decoding and CLAHE preprocessing |
| **NumPy** | Weight delta computation and DP noise generation |
| **aiohttp** | Async HTTP client for sending weights to mammo-global |
| **python-dotenv** | Environment variable management |

---

## Features

### 1. Privacy-First Training Pipeline
- Images are loaded one at a time into RAM using `io.BytesIO`
- Each image is trained on using `model.train_on_batch()` (1 gradient step)
- The image bytes and numpy arrays are deleted immediately (`del` + `gc.collect()`)
- **Zero disk writes** — no image is ever saved to disk at any point

### 2. Differential Privacy (Gaussian Mechanism)
Protects against **gradient inversion attacks**, where an attacker can mathematically reconstruct the original training images from the transmitted weight updates.

- **Step 1:** Computes the weight delta (change after training)
- **Step 2:** Clips the L2-norm of the delta to a sensitivity bound `S`
- **Step 3:** Adds calibrated Gaussian noise: `σ = S × √(2 × ln(1.25/δ)) / ε`
- **Result:** Provides formal `(ε, δ)`-differential privacy guarantee

Default settings: `ε = 1.0`, `δ = 1e-5` — strong privacy with minimal accuracy loss.

### 3. ZIP Dataset Support
- Accepts a single ZIP file containing any folder structure of JPEG/PNG images
- Extracts entirely in RAM (no temp files)
- Handles nested subfolders automatically (e.g., `dataset/images/scan.jpg`)
- Skips macOS metadata files (`__MACOSX`) automatically
- Hard limit: 500 images per upload, 500 MB ZIP size

### 4. Cryptographic Audit Trail
- After every training run, a SHA-256 hash of the final weight layer is computed
- This hash is the "privacy proof" — it proves training happened without revealing the images
- Full audit entry is appended to `audit_log.jsonl` (survives server restarts)
- Available via `GET /training-audit` for evaluator review

### 5. Mammogram Image Preprocessing (CLAHE)
- CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to every image
- This is the standard preprocessing pipeline for mammogram AI — it enhances local tissue contrast without amplifying noise
- Matches the preprocessing used in the original CBIS-DDSM training run

### 6. Heartbeat to mammo-global
- Background task pings mammo-global every 30 seconds
- Keeps the hospital node status as "Online" on the admin dashboard
- Silently handles connectivity failures (no crash if global server is offline)

---

## Security Features

### Endpoint Authentication
| Feature | Detail |
|---|---|
| X-API-Key | `/train-dataset` and `/training-audit` require a valid API key in the request header |
| Shared secret | Key must match `NODE_API_KEY` set in `.env` and `MAMMO_NODE_API_KEY` in mammo-global |

### Input Validation
| Feature | Detail |
|---|---|
| ZIP size limit | Hard 500 MB cap before any extraction begins |
| Image count limit | Maximum 500 images per training session |
| CORS whitelist | Only configured origins can make requests (not `*`) |
| File type check | Only `.jpg`, `.jpeg`, `.png` extensions accepted from ZIP contents |

### Differential Privacy
| Parameter | Default | Meaning |
|---|---|---|
| `DP_EPSILON` | `1.0` | Privacy budget — lower = more private, more noise |
| `DP_DELTA` | `1e-5` | Failure probability — effectively zero risk |
| `DP_L2_SENSITIVITY` | `1.0` | Maximum allowed weight change per image |

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/` | None | Health check — returns server status and model load state |
| `GET` | `/training-status` | None | Returns model accuracy metrics from last training run |
| `GET` | `/training-queue` | None | Returns current training queue size |
| `GET` | `/training-audit` | API Key | Full audit log of last training session (privacy proof) |
| `POST` | `/train-dataset` | API Key | Main training endpoint — accepts images or ZIP |
| `POST` | `/predict` | None | Run inference on a single mammogram image |

---

## Setup & Running

### Prerequisites
- Python 3.10+
- The trained model file: `mammo_v2.h5` (ResNet50 fine-tuned on CBIS-DDSM)
- Windows: Run with `--loop asyncio` flag (see below)

### Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
# Edit .env with your values

# Start the server
python -m uvicorn main:app --port 8000 --loop asyncio
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HOSPITAL_ID` | `HOSPITAL_01` | Unique identifier for this hospital node |
| `MODEL_PATH` | `./mammo_v2.h5` | Path to the trained ResNet50 model file |
| `METRICS_PATH` | `./model_metrics.json` | Path to model validation metrics |
| `GLOBAL_SERVER_URL` | `http://localhost:3001` | URL of the mammo-global coordinator |
| `NODE_API_KEY` | *(empty)* | API key for endpoint authentication |
| `ALLOWED_ORIGINS` | `http://localhost:3001,...` | Comma-separated CORS allowed origins |
| `AUDIT_LOG_PATH` | `./audit_log.jsonl` | Path to the append-only audit log |
| `DP_ENABLED` | `true` | Enable/disable Differential Privacy |
| `DP_EPSILON` | `1.0` | Privacy budget (ε) |
| `DP_DELTA` | `1e-5` | Failure probability (δ) |
| `DP_L2_SENSITIVITY` | `1.0` | Weight delta clip threshold |

---

## Model Information

The AI model (`mammo_v2.h5`) is a **ResNet50** architecture fine-tuned on the **CBIS-DDSM dataset** (Curated Breast Imaging Subset of DDSM) — a benchmark dataset of 10,556 mammogram images with expert annotations.

| Metric | Value |
|---|---|
| Base architecture | ResNet50 (ImageNet pretrained) |
| Dataset | CBIS-DDSM |
| Training samples | 10,556 mammograms |
| Validation accuracy | 92.1% |
| Input size | 224 × 224 × 3 |
| Output | Binary (Benign / Malignant) |

---

## Frequently Asked Questions

**Q: Where are the uploaded images stored?**
> They are never stored. Images are decoded into NumPy arrays in RAM, trained on, and immediately deleted. The `audit_log.jsonl` confirms this: `"images_written_to_disk": 0`.

**Q: Can someone reconstruct patient images from the weight updates?**
> This is prevented by Differential Privacy. Before any weight delta is transmitted, calibrated Gaussian noise is added. The mathematical guarantee is that even an adversary with unlimited computing power cannot distinguish between a model trained on Patient A's data vs. a model trained without it — to within the `ε` privacy budget.

**Q: What if the global server is unreachable?**
> Training completes normally. The weight delta transmission is attempted and fails silently. The audit log and SHA-256 hash are still generated locally. When the global server comes back online, the next training run will submit normally.

**Q: Why use `train_on_batch` instead of `model.fit`?**
> `train_on_batch` gives us per-image control. We can delete each image immediately after its gradient step, keeping memory usage constant regardless of dataset size. With `model.fit`, all images would need to be loaded simultaneously.
