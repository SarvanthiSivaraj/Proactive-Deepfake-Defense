from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.forensic.ml import VARIANT_BUILDERS
from src.preprocessing.loader import load_audio, save_audio
from src.verification.service import DeepfakeDefenseService


ATTACK_FOLDER_MAP = {
    "clean": "PROTECTED DERIVATIVE",
    "noise": "GAUSSIAN NOISE",
    "crop": "CROPPING",
    "lowpass": "LOWPASS FILTER",
    "resample": "RESAMPLING",
    "gain": "AMPLITUDE SCALING",
    "compression": "COMPRESSION",
    "unknown": "UNKNOWN MODIFICATION",
}


def _collect_sources() -> list[str]:
    candidates: list[str] = []
    for folder in ["input_audio", "data/eval_audio", "data/sample_audio"]:
        path = Path(folder)
        if not path.exists():
            continue
        for entry in sorted(path.iterdir()):
            if entry.is_file() and entry.suffix.lower() in {".wav", ".mp3", ".flac"}:
                candidates.append(str(entry))
    return candidates


def generate_dataset(dataset_dir: str, repeats: int = 3, progress_callback=None) -> list[dict[str, str]]:
    service = DeepfakeDefenseService()
    manifest: list[dict[str, str]] = []
    dataset_root = Path(dataset_dir)
    dataset_root.mkdir(parents=True, exist_ok=True)

    sources = _collect_sources()
    if not sources:
        raise RuntimeError("No source audio files were found for dataset generation.")

    # Calculate total files to generate: 1 clean + 7 attacks * repeats, per source file
    total_files = len(sources) * (1 + 7 * repeats)
    current_count = 0

    for source_path in sources:
        with tempfile.TemporaryDirectory() as temp_dir:
            protected_output = os.path.join(temp_dir, "protected.wav")
            embedded = service.embed_file(source_path, output_path=protected_output)
            protected_audio, sr = load_audio(embedded.output_path)
            source_name = Path(source_path).stem

            clean_dir = dataset_root / "clean"
            clean_dir.mkdir(parents=True, exist_ok=True)
            clean_path = clean_dir / f"{source_name}.wav"
            shutil.copyfile(embedded.output_path, clean_path)
            manifest.append({"file": str(clean_path), "label": "clean", "source": source_path})
            
            current_count += 1
            if progress_callback:
                progress_callback(current_count, total_files)

            for repeat_index in range(repeats):
                rng = np.random.default_rng(42 + repeat_index)
                for label, builder in VARIANT_BUILDERS.items():
                    if label == "clean":
                        continue
                    variant = builder(protected_audio, rng)
                    label_dir = dataset_root / label
                    label_dir.mkdir(parents=True, exist_ok=True)
                    variant_path = label_dir / f"{source_name}_{repeat_index}.wav"
                    save_audio(str(variant_path), variant, sr)
                    manifest.append({"file": str(variant_path), "label": label, "source": source_path})
                    
                    current_count += 1
                    if progress_callback:
                        progress_callback(current_count, total_files)

    manifest_path = dataset_root / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "label", "source"])
        writer.writeheader()
        writer.writerows(manifest)

    return manifest


def _label_order() -> list[str]:
    return [ATTACK_FOLDER_MAP[label] for label in ["clean", "noise", "crop", "lowpass", "resample", "gain", "compression", "unknown"]]


def run_benchmark(dataset_dir: str, output_dir: str, progress_callback=None) -> dict[str, object]:
    service = DeepfakeDefenseService()
    dataset_root = Path(dataset_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_dir)

    rows: list[dict[str, object]] = []
    y_true: list[str] = []
    y_pred_rule: list[str] = []
    y_pred_ml: list[str] = []
    ml_scores: list[dict[str, float]] = []

    wav_files = sorted(dataset_root.rglob("*.wav"))
    total_files = len(wav_files)
    current_count = 0

    for file_path in wav_files:
        expected_label = ATTACK_FOLDER_MAP.get(file_path.parent.name, "UNKNOWN MODIFICATION")
        tracemalloc.start()
        start = time.perf_counter()
        result = service.verify_file(str(file_path))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        ml_label = result.attack_analysis.get("likely_manipulation", "UNKNOWN")
        rule_label = result.attack_analysis.get("rule_label", ml_label)
        score_map = result.attack_analysis.get("scores", {})

        rows.append(
            {
                "file": str(file_path),
                "expected": expected_label,
                "ml_prediction": ml_label,
                "rule_prediction": rule_label,
                "final_authentication": result.final_authentication,
                "ber": result.ber,
                "ecc_success": result.ecc_success,
                "latency_ms": elapsed_ms,
                "memory_kb": peak / 1024.0,
            }
        )
        y_true.append(expected_label)
        y_pred_ml.append(ml_label)
        y_pred_rule.append(rule_label)
        ml_scores.append(score_map)

        current_count += 1
        if progress_callback:
            progress_callback(current_count, total_files)

    labels = _label_order()
    cm = confusion_matrix(y_true, y_pred_ml, labels=labels)
    report = classification_report(y_true, y_pred_ml, labels=labels, output_dict=True, zero_division=0)

    accuracy_ml = accuracy_score(y_true, y_pred_ml)
    accuracy_rule = accuracy_score(y_true, y_pred_rule)
    f1 = f1_score(y_true, y_pred_ml, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred_ml, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred_ml, average="weighted", zero_division=0)
    ber_mean = float(np.mean([row["ber"] for row in rows])) if rows else 0.0
    ecc_recovery_rate = float(np.mean([1.0 if row["ecc_success"] else 0.0 for row in rows])) if rows else 0.0
    latency_mean = float(np.mean([row["latency_ms"] for row in rows])) if rows else 0.0
    memory_mean = float(np.mean([row["memory_kb"] for row in rows])) if rows else 0.0

    os.makedirs(output_dir, exist_ok=True)
    rows_path = os.path.join(output_dir, "benchmark_rows.csv")
    summary_path = os.path.join(output_dir, "benchmark_summary.json")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    roc_path = os.path.join(output_dir, "roc_curves.png")

    with open(rows_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "samples": len(rows),
        "accuracy_ml": accuracy_ml,
        "accuracy_rule": accuracy_rule,
        "f1_weighted": f1,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "ber_mean": ber_mean,
        "ecc_recovery_rate": ecc_recovery_rate,
        "latency_ms_mean": latency_mean,
        "memory_kb_mean": memory_mean,
        "classification_report": report,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # Use premium dark background theme for matplotlib matching Streamlit console
    plt.style.use("dark_background")

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap="magma")
    plt.title("ML Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.colorbar()
    tick_positions = np.arange(len(labels))
    plt.xticks(tick_positions, labels, rotation=45, ha="right")
    plt.yticks(tick_positions, labels)
    
    # Annotate confusion matrix values in each cell
    thresh = cm.max() / 2.0 if cm.size > 0 else 1.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] < thresh else "black",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.savefig(cm_path, dpi=180, bbox_inches="tight")
    plt.close()

    if rows:
        y_bin = label_binarize(y_true, classes=labels)
        score_matrix = np.array([[score_map.get(label, 0.0) for label in labels] for score_map in ml_scores])
        plt.figure(figsize=(10, 8))
        for index, label in enumerate(labels):
            if np.unique(y_bin[:, index]).size < 2:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, index], score_matrix[:, index])
            auc_value = roc_auc_score(y_bin[:, index], score_matrix[:, index])
            plt.plot(fpr, tpr, label=f"{label} (AUC={auc_value:.3f})", linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1.5)
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("One-vs-Rest ROC Curves", fontsize=14, fontweight="bold", pad=15)
        plt.legend(fontsize=9, loc="lower right", framealpha=0.8)
        plt.grid(True, linestyle=":", alpha=0.3)
        plt.tight_layout()
        plt.savefig(roc_path, dpi=180, bbox_inches="tight")
        plt.close()

    return {
        "summary": summary,
        "rows_path": rows_path,
        "summary_path": summary_path,
        "confusion_matrix_path": cm_path,
        "roc_curves_path": roc_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a research-grade evaluation dataset and benchmark Phase 5.")
    parser.add_argument("--dataset-dir", default="dataset/generated", help="Where to generate and read the dataset")
    parser.add_argument("--output-dir", default="output/benchmarks", help="Where to write benchmark artifacts")
    parser.add_argument("--repeats", type=int, default=3, help="How many augmented copies to create per source file")
    parser.add_argument("--generate-only", action="store_true", help="Only generate the dataset")
    args = parser.parse_args()

    manifest = generate_dataset(args.dataset_dir, repeats=args.repeats)
    print(f"Generated {len(manifest)} samples in {args.dataset_dir}")
    if args.generate_only:
        return

    report = run_benchmark(args.dataset_dir, args.output_dir)
    print(json.dumps(report["summary"], indent=2))
    print("Confusion matrix:", report["confusion_matrix_path"])
    print("ROC curves:", report["roc_curves_path"])


if __name__ == "__main__":
    main()
