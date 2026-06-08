import os

from src.verification.service import DeepfakeDefenseService


INPUT_DIR = "input_audio"
OUTPUT_DIR = "output"


def verify_audio(filename):

    filepath = os.path.join(INPUT_DIR, filename)

    if not os.path.exists(filepath):

        print("\nFile not found:", filepath)

        return

    print("\nVERIFYING AUDIO")
    print("-------------------")

    service = DeepfakeDefenseService(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)
    result = service.verify_file(filepath)

    print()
    for line in result.report_lines:
        print(line, end="")

    report_path = os.path.join(OUTPUT_DIR, "verification_report.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.writelines(result.report_lines)

    print("\nReport saved:", report_path)


if __name__ == "__main__":
    filename = input("\nEnter audio filename: ")
    verify_audio(filename)
