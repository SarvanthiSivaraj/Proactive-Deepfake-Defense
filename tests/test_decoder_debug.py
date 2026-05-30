import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from src.preprocessing.loader import *
from src.preprocessing.stft import *

from src.encoder.qim import *
from src.decoder.qim_decoder import *


bits = [

    1,0,1,1,
    0,1,0,0
]

audio,sr = load_audio(

    "data/sample_audio/speech.wav"
)

stft = compute_stft(
    audio
)

mag,phase = split_mag_phase(
    stft
)

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

wm_mag,wm_phase = split_mag_phase(
    wm_stft
)

print("\nDEBUG")

for bit_i,group in enumerate(

        groups
):

    print(

        "\nBIT",

        bit_i,

        "EXPECTED",

        bits[bit_i]
    )

    votes=[]

    for row,col in group:

        value = wm_mag[
            row,
            col
        ]

        remainder = value % 0.5

        decoded = extract_qim(
            value
        )

        votes.append(
            decoded
        )

        print(

            "value=",
            round(float(value),4),

            "rem=",
            round(float(remainder),4),

            "dec=",
            decoded
        )

    print(

        "VOTES:",

        votes
    )