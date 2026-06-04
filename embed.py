import os

from src.verification.service import DeepfakeDefenseService


INPUT_DIR = "input_audio"
OUTPUT_DIR = "output"


def embed_audio(filename):

    filepath = os.path.join(INPUT_DIR, filename)

    if not os.path.exists(filepath):

        print("\nFile not found:", filepath)

        return

    print("\nEMBEDDING WATERMARK")
    print("-------------------")

    service = DeepfakeDefenseService(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)
    result = service.embed_file(filepath)

    print("\nDIRECT RAM BER:")
    print(round(float(result.direct_ber), 4))
    print("\nWatermarked audio saved to:", result.output_path)
    print("Copied to input for verification:", result.verification_ready_path)


if __name__ == "__main__":
    filename = input("\nEnter audio filename: ")
    embed_audio(filename)
