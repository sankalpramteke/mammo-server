# Mammo Server

FastAPI inference server for the **National Mammogram AI Detection System**.  
Runs locally at each hospital alongside `mammo-client`.

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ensure model file is present
# Place mammo_v2.h5 in the root folder (already done)

# 4. Run the server
uvicorn main:app --reload --port 8000
```

## Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/training-status` | FL training round info (polled by dashboard) |
| POST | `/predict` | Upload mammogram image → get AI prediction |
| GET | `/docs` | Auto-generated Swagger UI |

## Environment Variables (`.env`)

```
HOSPITAL_ID=AIIMS_NAGPUR
MODEL_PATH=./mammo_v2.h5
GLOBAL_SERVER_URL=http://localhost:9000
```

## Connection to mammo-client

`mammo-client` calls this server via `/api/predict` → `http://localhost:8000/predict`  
Both must run simultaneously for full functionality.
