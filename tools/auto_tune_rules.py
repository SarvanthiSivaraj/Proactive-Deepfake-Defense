import os
import csv
import json
import importlib.util
import sys


def load_tune_module():
    path = os.path.join(os.path.dirname(__file__), "..", "tests", "tune_rules.py")
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("tune_rules", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_audio_files(search_dirs):
    exts = (".wav", ".flac", ".mp3")
    files = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, fnames in os.walk(d):
            for f in fnames:
                if f.lower().endswith(exts):
                    files.append(os.path.join(root, f))
    return sorted(files)


def main():
    tr = load_tune_module()

    search_dirs = ["input_audio", os.path.join("data", "eval_audio"), os.path.join("data", "sample_audio")]
    audios = find_audio_files(search_dirs)
    if not audios:
        print("No audio files found in", search_dirs)
        return

    seeds = [42, 7, 101, 1234, 2026]

    os.makedirs("metadata", exist_ok=True)

    all_rows = []
    for seed in seeds:
        for a in audios:
            try:
                audio, sr = tr.load_audio(a)
            except Exception as e:
                print(f"Failed to load {a}: {e}")
                continue
            variants = tr.make_variants(audio, seed=seed)
            rows = tr.compute_metrics_for_variants(variants, sr)
            for r in rows:
                r["source"] = os.path.relpath(a)
                r["seed"] = int(seed)
                all_rows.append(r)

    if not all_rows:
        print("No variant metrics computed.")
        return

    # Save combined CSV
    csv_path = os.path.join("metadata", "variant_metrics_agg.csv")
    fieldnames = ["source", "seed", "attack", "expected"] + [k for k in all_rows[0].keys() if k not in ("source", "seed", "attack", "expected")]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Saved aggregated per-variant metrics to {csv_path}")

    print("Running aggregated grid search for rule thresholds (may take a few minutes)...")
    best = tr.grid_search(all_rows)
    print(f"Best aggregated rule accuracy: {best['score']:.3f}")

    out = {"best": best, "n_samples": len(all_rows), "n_seeds": len(seeds), "n_files": len(audios)}
    out_path = os.path.join("metadata", "rule_tuning_agg.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved aggregated tuning results to {out_path}")


if __name__ == "__main__":
    main()
