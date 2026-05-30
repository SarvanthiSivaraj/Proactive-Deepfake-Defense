import sys
import os
import zlib

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

from src.security.signature import *

from src.evaluation.metrics import *


# ---------- STRONGER ECC ----------

rsc = RSCodec(
    128
)

# ---------- METADATA ----------

metadata = generate_metadata()

print("\nOriginal Metadata:")
print(metadata)

# ---------- SERIALIZE ----------

metadata_bytes = serialize_payload(
    metadata
)

# ---------- SIGN ----------

signature = sign_payload(
    metadata_bytes
)

print("\nSHA256 Signature:")
print(signature)

# ---------- PACKET ----------

signed_packet = {

    "metadata": metadata,

    "signature": signature
}

packet_bytes = serialize_payload(
    signed_packet
)

print("\nRaw Packet Length:")
print(
    len(packet_bytes)
)

# ---------- COMPRESS ----------

compressed_packet = zlib.compress(
    packet_bytes
)

print("\nCompressed Length:")
print(
    len(compressed_packet)
)

# ---------- ECC ----------

ecc_bytes = rsc.encode(
    compressed_packet
)

print("\nECC Length:")
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

ber = compute_ber(

    payload_bits,

    recovered_bits
)

print("\nBER:")
print(
    ber
)

# ---------- BITS→BYTES ----------

recovered_bytes = bits_to_bytes(
    recovered_bits
)

# ---------- ECC DECODE ----------

print("\nAttempting ECC Decode...")

try:

    decoded_compressed = rsc.decode(

        recovered_bytes

    )[0]

    decoded_packet = zlib.decompress(

        decoded_compressed
    )

    recovered_packet = deserialize_payload(

        decoded_packet
    )

    print("\nRecovered Packet:")
    print(
        recovered_packet
    )

    recovered_metadata = recovered_packet[
        "metadata"
    ]

    recovered_signature = recovered_packet[
        "signature"
    ]

    verified = verify_signature(

        serialize_payload(
            recovered_metadata
        ),

        recovered_signature
    )

    print("\nSignature Verification:")
    print(
        verified
    )

except Exception as e:

    print("\nPIPELINE FAILED:")
    print(e)