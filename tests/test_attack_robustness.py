import sys
import os

sys.path.append(
    os.path.abspath(".")
)

import numpy as np
from reedsolo import RSCodec

from src.preprocessing.loader import *
from src.preprocessing.stft import *

from src.encoder.qim import *
from src.decoder.qim_decoder import *

from src.payload.bitstream import *

from src.evaluation.metrics import *


rsc = RSCodec(16)

payload = b"HELLO"

print("\nOriginal Payload:")
print(payload)

ecc_bytes = rsc.encode(
    payload
)

payload_bits = bytes_to_bits(
    ecc_bytes
)

audio,sr = load_audio(

    "data/sample_audio/speech.wav"
)

stft = compute_stft(
    audio
)

mag,phase = split_mag_phase(
    stft
)

embedded_mag,groups = embed_payload_qim(

    mag,

    payload_bits,

    seed=42
)

modified = merge_mag_phase(

    embedded_mag,

    phase
)

watermarked = inverse_stft(
    modified
)

# ---------- ATTACK ----------

noise_std = 0.002

attacked = watermarked + np.random.normal(

    0,

    noise_std,

    len(watermarked)
)

# ----------------------------

attack_stft = compute_stft(
    attacked
)

attack_mag,_ = split_mag_phase(
    attack_stft
)

recovered_bits = extract_payload_qim(

    attack_mag,

    groups
)

ber = compute_ber(

    payload_bits,

    recovered_bits
)

print("\nBER:")
print(ber)

recovered_bytes = bits_to_bytes(

    recovered_bits
)

try:

    decoded = rsc.decode(
        recovered_bytes
    )

    print("\nECC Success:")
    print(decoded[0])

except Exception as e:

    print("\nECC FAILED:")
    print(e)