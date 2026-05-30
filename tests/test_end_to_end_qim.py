import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from src.preprocessing.loader import *
from src.preprocessing.stft import *

from src.encoder.qim import *
from src.decoder.qim_decoder import *

from src.evaluation.metrics import *


payload = [

    1,0,1,1,

    0,0,1,1
]


# LOAD AUDIO

audio, sr = load_audio(

    "data/sample_audio/speech.wav"
)


# STFT

stft = compute_stft(

    audio
)

mag, phase = split_mag_phase(

    stft
)


# EMBED PAYLOAD

embedded_mag, rows = embed_payload_qim(

    mag,

    payload
)

print("\nSelected Rows:")

print(rows)


# DEBUG — EMBEDDED VALUES

print("\nEmbedded Values:")

for row in rows:

    print(

        embedded_mag[

            row,

            100
        ]
    )


# RECONSTRUCT AUDIO

modified_stft = merge_mag_phase(

    embedded_mag,

    phase
)

watermarked = inverse_stft(

    modified_stft
)


# SAVE

save_audio(

    "data/sample_audio/e2e_watermarked.wav",

    watermarked,

    sr
)


# RELOAD AUDIO

wm_audio, sr = load_audio(

    "data/sample_audio/e2e_watermarked.wav"
)


# NEW STFT

wm_stft = compute_stft(

    wm_audio
)

wm_mag, wm_phase = split_mag_phase(

    wm_stft
)


# DEBUG — RELOADED VALUES

print("\nReloaded Values:")

for row in rows:

    print(

        wm_mag[

            row,

            100
        ]
    )


# RECOVER

recovered = extract_payload_qim(

    wm_mag,

    rows
)


print("\nOriginal Payload:")

print(payload)

print("\nRecovered Payload:")

print(recovered)


# BER

ber = compute_ber(

    payload,

    recovered
)

print("\nBER:")

print(ber)