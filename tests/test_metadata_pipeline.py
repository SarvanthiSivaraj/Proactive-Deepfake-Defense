import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from reedsolo import RSCodec

from src.preprocessing.loader import *
from src.preprocessing.stft import *

from src.encoder.qim import *
from src.decoder.qim_decoder import *

from src.payload.metadata import *
from src.payload.serialize import *
from src.payload.bitstream import *

from src.evaluation.metrics import *


# ---------- ECC ----------

rsc = RSCodec(
    64
)

# ---------- GENERATE METADATA ----------

metadata = generate_metadata()

print("\nOriginal Metadata:")
print(metadata)

# ---------- SERIALIZE ----------

payload_bytes = serialize_payload(
    metadata
)

print("\nSerialized Payload:")
print(payload_bytes)

print("\nPayload Byte Length:")
print(
    len(payload_bytes)
)

# ---------- ECC ENCODE ----------

ecc_bytes = rsc.encode(
    payload_bytes
)

print("\nECC Byte Length:")
print(
    len(ecc_bytes)
)

payload_bits = bytes_to_bits(
    ecc_bytes
)

print("\nPayload Bits:")
print(
    len(payload_bits)
)

# ---------- LOAD AUDIO ----------

audio,sr = load_audio(

    "data/sample_audio/speech.wav"
)

stft = compute_stft(
    audio
)

mag,phase = split_mag_phase(
    stft
)

# ---------- EMBED ----------

embedded_mag,groups = embed_payload_qim(

    mag,

    payload_bits,

    seed=42
)

# ---------- ROUNDTRIP ----------

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

wm_mag,_ = split_mag_phase(
    wm_stft
)

# ---------- EXTRACT ----------

recovered_bits = extract_payload_qim(

    wm_mag,

    groups
)

# ---------- BER ----------

ber = compute_ber(

    payload_bits,

    recovered_bits
)

print("\nBER:")
print(
    ber
)

# ---------- BITS → BYTES ----------

recovered_bytes = bits_to_bytes(

    recovered_bits
)

print("\nRecovered Byte Length:")
print(
    len(recovered_bytes)
)

# ---------- ECC DECODE ----------

print("\nAttempting ECC Decode...")

try:

    decoded_bytes = rsc.decode(

        recovered_bytes

    )[0]

    recovered_metadata = deserialize_payload(

        decoded_bytes
    )

    print("\nRecovered Metadata:")
    print(
        recovered_metadata
    )

except Exception as e:

    print("\nECC Decode Failed:")
    print(e)