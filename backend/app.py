from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .runtime_model_runner import generate_job_id, score_csv_file

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
TMP_DIR = BASE_DIR / "tmp"
UPLOAD_DIR = TMP_DIR / "uploads"
PRED_DIR = TMP_DIR / "predictions"
SUMMARY_DIR = TMP_DIR / "summaries"

for folder in [TMP_DIR, UPLOAD_DIR, PRED_DIR, SUMMARY_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Fraud Detection Web", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    paysim_weight: float = 0.3,
    kartik_weight: float = 0.7,
    final_threshold: float = 0.5,
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    job_id = generate_job_id()
    upload_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    with upload_path.open("wb") as buffer:
        buffer.write(await file.read())

    prediction_path = PRED_DIR / f"{job_id}.csv"

    try:
        _, summary = score_csv_file(
            upload_path,
            prediction_path,
            paysim_weight=paysim_weight,
            kartik_weight=kartik_weight,
            final_threshold=final_threshold,
        )
    except Exception as exc:  # broad so we can return a clean error
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    summary.download_path = str(prediction_path)
    summary_json = summary.to_json()
    summary_json["download_url"] = f"/api/jobs/{job_id}/download"

    with (SUMMARY_DIR / f"{job_id}.json").open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    return {"job_id": job_id, "summary": summary_json}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    summary_file = SUMMARY_DIR / f"{job_id}.json"
    if not summary_file.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    with summary_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str):
    prediction_path = PRED_DIR / f"{job_id}.csv"
    if not prediction_path.exists():
        raise HTTPException(status_code=404, detail="Prediction file not found")
    return FileResponse(
        prediction_path,
        media_type="text/csv",
        filename=f"predictions_{job_id}.csv",
    )


# --------------- FRONTEND ROUTES ---------------


def _send_page(name: str):
    page_path = FRONTEND_DIR / name
    if not page_path.exists():
        raise HTTPException(status_code=404, detail="Page missing")
    return FileResponse(page_path)


@app.get("/")
def landing_page():
    return _send_page("index.html")


@app.get("/upload")
def upload_page():
    return _send_page("upload.html")


@app.get("/results")
def results_page():
    return _send_page("results.html")

