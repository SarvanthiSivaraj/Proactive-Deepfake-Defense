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

stft = compute_stft(

    audio
)

mag, phase = split_mag_phase(

    stft
)


row = 100
col = 50

original = mag[

    row,

    col
]

bit = 1

mag[

    row,

    col

] = embed_qim(

        original,

        bit
)


modified_stft = merge_mag_phase(

    mag,

    phase
)

watermarked = inverse_stft(

    modified_stft
)


save_audio(

    "data/sample_audio/watermarked.wav",

    watermarked,

    sr
)

print(

    "\nWatermarked audio saved."
)