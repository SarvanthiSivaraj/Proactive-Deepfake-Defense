import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from src.preprocessing.loader import *
from src.preprocessing.stft import *

audio, sr = load_audio(
    "data/sample_audio/speech.wav"
)

stft = compute_stft(audio)

mag, phase = split_mag_phase(stft)

r = 203
c = 312

print("\nORIGINAL:")
print(mag[r,c])

mag[r,c] += 0.5

modified = merge_mag_phase(
    mag,
    phase
)

watermarked = inverse_stft(
    modified
)

wm_stft = compute_stft(
    watermarked
)

wm_mag, wm_phase = split_mag_phase(
    wm_stft
)

print("\nAFTER ROUNDTRIP:")
print(wm_mag[r,c])

print("\nDRIFT:")
print(
    wm_mag[r,c] -
    mag[r,c]
)