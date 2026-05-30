import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from src.preprocessing.loader import *
from src.preprocessing.stft import *

from src.payload.metadata import *
from src.payload.serialize import *
from src.payload.bitstream import *

from src.crypto.signer import *

from src.ecc.encode import *

from src.encoder.qim import *
from src.decoder.qim_decoder import *

from src.evaluation.metrics import *


# --------------------
# METADATA
# --------------------

metadata = generate_metadata()

payload = serialize_payload(
    metadata
)

print("\nMetadata:")
print(metadata)


# --------------------
# SIGNATURE
# --------------------

private_key, public_key = generate_keys()

signature = sign_message(

    private_key,

    payload
)

print("\nSignature Length:")
print(len(signature))


# --------------------
# ECC
# --------------------

protected = rs_encode(

    signature
)


# --------------------
# BYTES → BITS
# --------------------

bits = bytes_to_bits(

    protected
)

print("\nPayload Bit Length:")
print(len(bits))


# FIRST SMALL TEST

bits = bits[:32]


# --------------------
# AUDIO LOAD
# --------------------

audio, sr = load_audio(

    "data/sample_audio/speech.wav"
)


# --------------------
# STFT
# --------------------

stft = compute_stft(

    audio
)

mag, phase = split_mag_phase(

    stft
)


# --------------------
# EMBEDDING
# --------------------

embedded_mag, rows = embed_payload_qim(

    mag,

    bits
)

print("\nSelected Locations:")
print(rows)


# --------------------
# RECONSTRUCT AUDIO
# --------------------

modified = merge_mag_phase(

    embedded_mag,

    phase
)

watermarked = inverse_stft(

    modified
)


save_audio(

    "data/sample_audio/real_pipeline.wav",

    watermarked,

    sr
)


# --------------------
# RELOAD AUDIO
# --------------------

wm_audio, sr = load_audio(

    "data/sample_audio/real_pipeline.wav"
)


wm_stft = compute_stft(

    wm_audio
)

wm_mag, wm_phase = split_mag_phase(

    wm_stft
)


# --------------------
# RECOVERY
# --------------------

recovered = extract_payload_qim(

    wm_mag,

    rows
)


# --------------------
# RESULTS
# --------------------

print("\nOriginal Bits:")
print(bits)

print("\nRecovered Bits:")
print(recovered)


ber = compute_ber(

    bits,

    recovered
)

print("\nBER:")
print(ber)