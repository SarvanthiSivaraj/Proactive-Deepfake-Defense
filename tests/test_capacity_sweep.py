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


sizes = [

    8,

    16,

    32,

    64,

    128
]


audio, sr = load_audio(

    "data/sample_audio/speech.wav"
)

stft = compute_stft(audio)

mag, phase = split_mag_phase(stft)


for n in sizes:

    bits = [

        i % 2

        for i in range(n)
    ]

    embedded_mag, locations = embed_payload_qim(

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

    save_audio(

        "data/sample_audio/temp.wav",

        watermarked,

        sr
    )

    wm_audio, sr = load_audio(

        "data/sample_audio/temp.wav"
    )

    wm_stft = compute_stft(

        wm_audio
    )

    wm_mag, wm_phase = split_mag_phase(

        wm_stft
    )

    recovered = extract_payload_qim(

        wm_mag,

        locations
    )

    ber = compute_ber(

        bits,

        recovered
    )

    print(

        f"\nPayload={n} bits"
    )

    print(

        f"BER={ber}"
    )