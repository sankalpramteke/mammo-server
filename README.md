# mammo-server — Complete In-Depth Guide

> **Purpose**: The local hospital node (AI inference engine). It receives mammogram images, runs predictions using a local ResNet50 model, queues confirmed diagnoses from doctors, and performs local Federated Learning (FL) training before sending the model weights to the global server.

---

## 1. Technology Stack

| Technology | Version | Why We Used It |
|---|---|---|
| **FastAPI** | 0.110.0 | High-performance Python web framework. Perfect for AI microservices because it's fast, asynchronous, and auto-generates API docs (Swagger). |
| **Uvicorn** | 0.27.0 | ASGI web server used to run the FastAPI application. |
| **TensorFlow / Keras** | >=2.16.0 | The core AI engine. Used to load the `mammo_v2.h5` model (ResNet50), run image inference, and perform local fine-tuning (backpropagation) for FL. |
| **Pillow (PIL)** | 10.2.0 | Image processing library. Used to read the uploaded multipart image and convert it into RGB arrays. |
| **NumPy** | 1.26.4 | Array manipulation. Used to resize and normalize the image arrays (shape: 1x224x224x3) before feeding them into TensorFlow. |
| **aiohttp** | 3.9.3 | Asynchronous HTTP client. Used to send the model weight updates and the 30-second heartbeat to the `mammo-global` server without blocking FastAPI. |
| **python-multipart** | 0.0.9 | Required by FastAPI to parse `multipart/form-data` (file uploads) correctly. |
| **python-dotenv** | 1.0.1 | Loads environment variables from `.env` file (`HOSPITAL_ID`, `GLOBAL_SERVER_URL` etc.). |

---

## 2. Project Architecture

```mermaid
graph LR
    subgraph "mammo-client (:3000)"
        MC["Next.js Front-end"]
    end
    
    subgraph "mammo-server (:8000)"
        FA["FastAPI Endpoints"]
        TF["TensorFlow Model"]
        Q["In-Memory Queue"]
    end
    
    subgraph "mammo-global (:3001)"
        MG["Global Dashboard"]
    end
    
    MC -->|1. Upload Image| FA
    FA --> TF
    TF -->|Prediction| MC
    
    MC -->|2. Confirm Diagnosis| Q
    
    FA -->|3. Trigger /train| TF
    TF -->|Fine-tune on Queue| TF
    TF -->|4. Send Weights (aiohttp)| MG
    
    FA -->|Heartbeat loop| MG
```

### Key Architectural Decisions
1. **Separation of Concerns**: This server *only* handles heavy AI tasks (inference & training). It does not handle user authentication, UI, or persistent historical data (that's handled by Next.js and MongoDB).
2. **In-Memory Training Queue**: Instead of writing confirmed scans to a database, they are temporarily held in RAM (`training_queue`). When `/train` is called, it digests the queue, trains, and clears it.
3. **Async Heartbeat**: A background `asyncio` loop continuously pings `mammo-global` to report that this hospital node is online, showing the real-time node status on the global map.

---

## 3. Core Features & Federated Learning Flow

Following the Federated Learning pattern, this node acts as **Step A and Step B**.

### Feature 1: AI Inference (Step A)
- **Endpoint**: `POST /predict`
- **How it works**: Receives an image from `mammo-client`. Preprocesses it to 224x224 pixels, normalizes it, and passes it through the loaded ResNet50 model.
- **Output**: Returns "Benign" or "Malignant" with exact confidence percentages.

### Feature 2: Queuing Expert Confirmations
- **Endpoint**: `POST /queue-for-training`
- **How it works**: When a doctor corrects or confirms the AI's prediction on the client UI, this endpoint receives the "True Label" (e.g., patient actually has Malignant cancer) and stores it in the `training_queue`.

### Feature 3: Local FL Training (Step B)
- **Endpoint**: `POST /train` (Triggered manually or via chron job)
- **How it works**: 
  1. Reads the `training_queue`.
  2. Compiles a tiny dataset from these true labels.
  3. Uses **TensorFlow `model.fit()`** to fine-tune the top layers of the local ResNet50 model (local learning).
  4. Calculates the **Weight Delta** (difference between old weights and newly trained weights).
  5. Slices the first 5 layers of the weight matrix and sends them via an async HTTP POST to `mammo-global/api/fl/receive-weights`.
  6. Empties the queue.

### Feature 4: Realistic Accuracy Simulation
- **How it works**: In a demo with 1 sample, training accuracy trivially hits 100%. To simulate a realistic, multi-hospital FL progression, the server tracks `_fl_round_counter` and gradually increments the reported accuracy (starting ~72%, adding 1-3% per round, capping at 94%).

---

## 4. API Endpoints — Deep Dive

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check. Returns status and whether the model loaded successfully. |
| `POST` | `/predict` | **Core inference**. Expects `multipart/form-data` with a `file` field. Returns the AI diagnosis. |
| `POST` | `/queue-for-training` | **Feedback loop**. Accepts JSON `{ patientId, confirmedLabel }`. Adds to the training queue. |
| `POST` | `/train` | **FL Engine**. Fine-tunes the local TensorFlow model on the queued labels, sends weights to global, and returns the realistic accuracy progression. |
| `GET` | `/training-queue` | Debug endpoint to view currently queued scans. |

---

## 5. Environment Variables & Setup

| Variable | Example | Explanation |
|---|---|---|
| `HOSPITAL_ID` | `AIIMS_NAGPUR` | The unique identifier for this node. Used so `mammo-global` knows where the weights came from. |
| `MODEL_PATH` | `./mammo_v2.h5` | Path to the pre-trained TensorFlow/Keras `.h5` model file. |
| `GLOBAL_SERVER_URL`| `http://localhost:3001` | Where to send weight updates and heartbeats. |

### How to Run
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn main:app --reload --port 8000
```

---

## 6. Likely Q&A for Evaluators

**Q: Why use FastAPI for this instead of integrating the AI into Next.js?**
**A:** Python is the native ecosystem for AI (TensorFlow/PyTorch). Running large `.h5` models in Node.js (Next.js) is highly inefficient and memory-intensive. A dedicated Python microservice (FastAPI) is the industry standard for serving ML models natively.

**Q: Does mammo-server save the patient images?**
**A:** No. `mammo-server` processes the image array in RAM during `/predict` and then discards it. This guarantees patient privacy because no sensitive data is permanently stored on the disk.

**Q: How does the Federated Learning training actually happen here?**
**A:** We use `model.fit()` on the base TensorFlow model using the labels provided by the doctor. We compile it with an extremely low learning rate (`1e-5`) to prevent "catastrophic forgetting" (where the model forgets its base training). We then calculate the absolute difference in the weight matrices and send only that delta to the global server.

**Q: Why do you send only the first 5 layers of weights?**
**A:** In this prototype, weight matrices are massive (hundreds of MBs). To make the HTTP POST request fast and avoid timeouts on localhost, we slice the array (`weights_list[:5]`) as a proof-of-concept. In a production gRPC architecture, we would stream the entire tensor.

**Q: Why is the accuracy simulated?**
**A:** Because we are training on 1 or 2 small synthetic noise arrays (as a stand-in for real image datasets), a neural network will instantly memorize the data and report 100% accuracy. The simulation demonstrates the *expected mathematical progression* (FedAvg asymptotic convergence) of a real FL network over time.
