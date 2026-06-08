from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.verification.service import DeepfakeDefenseService


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}


@dataclass
class BatchFileResult:
    file: str
    status: str
    attack: str
    confidence: str
    authentication: str
    provenance: dict[str, Any]
    metrics: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BatchVerifier:
    def __init__(self, service: DeepfakeDefenseService | None = None, output_dir: str = "output/batch_reports"):
        self.service = service or DeepfakeDefenseService()
        self.output_dir = output_dir

    def collect_files(self, folder_path: str, recursive: bool = True) -> list[str]:
        root = Path(folder_path)
        if not root.exists():
            raise FileNotFoundError(folder_path)

        pattern = "**/*" if recursive else "*"
        files = []
        for path in root.glob(pattern):
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                files.append(str(path))
        return sorted(files)

    def verify_files(self, file_paths: Iterable[str]) -> dict[str, Any]:
        rows: list[BatchFileResult] = []
        for file_path in file_paths:
            try:
                result = self.service.verify_file(file_path)
                rows.append(
                    BatchFileResult(
                        file=file_path,
                        status=result.auth_card_status,
                        attack=result.attack_analysis.get("likely_manipulation", "UNKNOWN"),
                        confidence=result.attack_analysis.get("confidence", "LOW"),
                        authentication=result.final_authentication,
                        provenance=result.provenance,
                        metrics=result.metrics,
                    )
                )
            except Exception as exc:
                rows.append(
                    BatchFileResult(
                        file=file_path,
                        status="ERROR",
                        attack="ERROR",
                        confidence="LOW",
                        authentication="ERROR",
                        provenance={},
                        metrics={},
                        error=str(exc),
                    )
                )

        summary = self._summarize(rows)
        csv_path, json_path = self._write_reports(rows, summary)
        return {
            "summary": summary,
            "files": [row.to_dict() for row in rows],
            "csv_path": csv_path,
            "json_path": json_path,
        }

    def verify_folder(self, folder_path: str, recursive: bool = True) -> dict[str, Any]:
        file_paths = self.collect_files(folder_path, recursive=recursive)
        return self.verify_files(file_paths)

    def _summarize(self, rows: list[BatchFileResult]) -> dict[str, Any]:
        summary = {
            "total": len(rows),
            "authentic": 0,
            "protected_derivative": 0,
            "not_authentic": 0,
            "errors": 0,
        }
        for row in rows:
            if row.status == "AUTHENTIC":
                summary["authentic"] += 1
            elif row.status == "PROTECTED DERIVATIVE":
                summary["protected_derivative"] += 1
            elif row.status == "NOT AUTHENTIC":
                summary["not_authentic"] += 1
            else:
                summary["errors"] += 1
        return summary

    def _write_reports(self, rows: list[BatchFileResult], summary: dict[str, Any]) -> tuple[str, str]:
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, "batch_verification_report.csv")
        json_path = os.path.join(self.output_dir, "batch_verification_report.json")

        with open(csv_path, "w", newline="", encoding="utf-8") as csv_handle:
            writer = csv.DictWriter(
                csv_handle,
                fieldnames=["file", "status", "attack", "confidence", "authentication", "provenance", "metrics", "error"],
            )
            writer.writeheader()
            for row in rows:
                serialized = row.to_dict()
                serialized["provenance"] = json.dumps(serialized["provenance"], ensure_ascii=True)
                serialized["metrics"] = json.dumps(serialized["metrics"], ensure_ascii=True)
                writer.writerow(serialized)

        with open(json_path, "w", encoding="utf-8") as json_handle:
            json.dump({"summary": summary, "files": [row.to_dict() for row in rows]}, json_handle, indent=2)

        return csv_path, json_path
