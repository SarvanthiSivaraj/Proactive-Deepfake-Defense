import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from src.preprocessing.loader import *
from src.preprocessing.stft import *

from src.encoder.qim import *


audio, sr = load_audio(
    "data/sample_audio/speech.wav"
)

stft = compute_stft(audio)

mag, phase = split_mag_phase(
    stft
)

bits=[1,0,1,0]

embedded_mag, groups = embed_payload_qim(
    mag,
    bits
)

modified = merge_mag_phase(
    embedded_mag,
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

print("\nSURVIVAL DEBUG\n")

for bit_i,group in enumerate(groups):

    print(
        f"\nBIT {bit_i}  expected={bits[bit_i]}"
    )

    for row,col in group:

        before = embedded_mag[
            row,
            col
        ]

        after = wm_mag[
            row,
            col
        ]

        print(
            f"({row},{col})"
        )

        print(
            "embedded =",before
        )

        print(
            "after     =",after
        )

        print(
            "drift     =",
            after-before
        )