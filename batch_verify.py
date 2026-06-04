from __future__ import annotations

import argparse
import json

from src.verification.batch import BatchVerifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch verification for a folder of audio files.")
    parser.add_argument("folder", help="Folder containing wav/mp3/flac files")
    parser.add_argument("--no-recursive", action="store_true", help="Only scan the top level of the folder")
    parser.add_argument("--output-dir", default="output/batch_reports", help="Directory for CSV/JSON reports")
    args = parser.parse_args()

    verifier = BatchVerifier(output_dir=args.output_dir)
    report = verifier.verify_folder(args.folder, recursive=not args.no_recursive)

    print(json.dumps(report["summary"], indent=2))
    print("CSV:", report["csv_path"])
    print("JSON:", report["json_path"])


if __name__ == "__main__":
    main()
