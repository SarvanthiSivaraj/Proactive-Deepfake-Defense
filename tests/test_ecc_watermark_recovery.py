import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from src.preprocessing.loader import *
from src.preprocessing.stft import *

from src.payload.bitstream import *

from src.ecc.encode import *
from src.ecc.decode import *

from src.encoder.qim import *
from src.decoder.qim_decoder import *

from src.evaluation.metrics import *


# ------------------
# ORIGINAL PAYLOAD
# ------------------

payload = b"HELLO"

print("\nOriginal Payload:")

print(payload)


# ------------------
# ECC ENCODE
# ------------------

protected = rs_encode(

    payload
)

print("\nECC Bytes:")

print(protected)


# ------------------
# BYTES → BITS
# ------------------

bits = bytes_to_bits(

    protected
)

print("\nTotal ECC Bits:")

print(len(bits))


# SMALLER EXPERIMENT

bits = bits[:80]


# ------------------
# LOAD AUDIO
# ------------------

audio, sr = load_audio(

    "data/sample_audio/speech.wav"
)


# ------------------
# STFT
# ------------------

stft = compute_stft(

    audio
)

mag, phase = split_mag_phase(

    stft
)


# ------------------
# EMBED
# ------------------

embedded_mag, rows = embed_payload_qim(

    mag,

    bits
)

print("\nEmbedding Locations:")

print(rows)


# ------------------
# RECONSTRUCT
# ------------------

modified = merge_mag_phase(

    embedded_mag,

    phase
)

watermarked = inverse_stft(

    modified
)


save_audio(

    "data/sample_audio/ecc_recovery.wav",

    watermarked,

    sr
)


# ------------------
# RELOAD AUDIO
# ------------------

wm_audio, sr = load_audio(

    "data/sample_audio/ecc_recovery.wav"
)

wm_stft = compute_stft(

    wm_audio
)

wm_mag, wm_phase = split_mag_phase(

    wm_stft
)


# ------------------
# RECOVER BITS
# ------------------

recovered_bits = extract_payload_qim(

    wm_mag,

    rows
)

print("\nRecovered Bits:")

print(recovered_bits)


# ------------------
# BER
# ------------------

ber = compute_ber(

    bits,

    recovered_bits
)

print("\nBER:")

print(ber)


# ------------------
# BITS → BYTES
# ------------------

recovered_bytes = bits_to_bytes(

    recovered_bits
)

print("\nRecovered Bytes:")

print(recovered_bytes)

print("\nRecovered Byte Length:")

print(

    len(recovered_bytes)
)


# ------------------
# ECC DECODE
# ------------------

print("\nAttempting ECC Decode...")


try:

    if len(

            recovered_bytes

    ) >= 10:

        decoded = rs_decode(

            recovered_bytes
        )

        print("\nECC Success:")

        print(decoded)

    else:

        print(

            "\nInsufficient bytes for ECC decode."
        )

except Exception as e:

    print("\nECC Decode Failed:")

    print(e)