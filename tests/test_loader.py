import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from src.preprocessing.loader import (
    load_audio,
    normalize_audio,
    save_audio
)
audio, sr = load_audio(
    "data/sample_audio/speech.wav"
)

print("Loaded.")

print("Sample Rate:", sr)

print("Shape:", audio.shape)

audio = normalize_audio(audio)

save_audio(
    "data/sample_audio/output.wav",
    audio,
    sr
)

print("Saved.")