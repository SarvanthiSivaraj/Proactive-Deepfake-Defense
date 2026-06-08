from __future__ import annotations

import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.verification.batch import BatchVerifier
from src.verification.service import DeepfakeDefenseService


app = FastAPI(title="Proactive Deepfake Defense API", version="5.0.0")
service = DeepfakeDefenseService()
batch_verifier = BatchVerifier(service=service)


def _persist_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(upload.file.read())
        return handle.name


@app.get("/")
def root():
    return {"status": "ok", "service": "Proactive Deepfake Defense API"}


@app.post("/embed")
def embed(upload: UploadFile = File(...)):
    temp_path = _persist_upload(upload)
    try:
        result = service.embed_file(temp_path)
        return {
            "watermarked_audio": result.watermarked_audio_b64,
            "metadata": result.metadata,
            "output_path": result.output_path,
            "verification_ready_path": result.verification_ready_path,
            "source_hash": result.source_hash,
            "payload_bits": result.payload_bits,
            "direct_ber": result.direct_ber,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/verify")
def verify(upload: UploadFile = File(...)):
    temp_path = _persist_upload(upload)
    try:
        result = service.verify_file(temp_path)
        return {
            "authentication": {
                "card_status": result.auth_card_status,
                "final": result.final_authentication,
                "color": result.auth_card_color,
            },
            "attack_analysis": result.attack_analysis,
            "provenance": result.provenance,
            "metrics": result.metrics,
            "ber": result.ber,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/batch_verify")
def batch_verify(files: list[UploadFile] = File(...)):
    temp_paths = []
    try:
        for upload in files:
            temp_paths.append(_persist_upload(upload))
        report = batch_verifier.verify_files(temp_paths)
        return report
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run("api.fastapi_server:app", host="0.0.0.0", port=8000, reload=False)
