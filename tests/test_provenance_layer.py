import os
import sys

import numpy as np

sys.path.append(
    os.path.abspath(".")
)

from src.payload.metadata import *
from src.payload.serialize import *
from src.security.signature import *


def provenance_decision(signature_valid, source_hash_match):

    if signature_valid and source_hash_match:
        return "AUTHENTIC PROTECTED SOURCE MATCH"

    if signature_valid:
        return "AUTHENTIC PROTECTED DERIVATIVE"

    return "NOT AUTHENTIC"


base_audio = np.array(
    [0.0, 0.25, -0.5, 0.75, -1.0, 0.5, 0.125],
    dtype=np.float32
)

modified_audio = base_audio.copy()
modified_audio[2] += 0.01

different_audio = np.array(
    [1.0, -0.75, 0.5, -0.25, 0.0, 0.25, -0.5],
    dtype=np.float32
)

source_hash = compute_audio_hash(
    base_audio,
    sample_rate=44100
)

same_hash = compute_audio_hash(
    base_audio,
    sample_rate=44100
)

modified_hash = compute_audio_hash(
    modified_audio,
    sample_rate=44100
)

different_hash = compute_audio_hash(
    different_audio,
    sample_rate=44100
)

assert source_hash == same_hash
assert source_hash != modified_hash
assert source_hash != different_hash

metadata = generate_metadata(
    source_hash=source_hash,
    timestamp="2026-06-01 18:05:23",
    audio_id="AUDIO001",
    generator="VOICE_GEN_V1",
    creator="Sarvanthikha",
    organization="Amrita University",
    model_version="ProactiveDefense-v1.0"
)

expected_keys = {
    "id",
    "generator",
    "timestamp",
    "creator",
    "organization",
    "model_version",
    "source_hash"
}

assert set(metadata.keys()) == expected_keys
assert metadata["source_hash"] == source_hash
assert metadata["creator"] == "Sarvanthikha"
assert metadata["organization"] == "Amrita University"
assert metadata["model_version"] == "ProactiveDefense-v1.0"

payload_bytes = serialize_payload(
    metadata
)

signature = sign_payload(
    payload_bytes
)

signature_valid = verify_signature(
    payload_bytes,
    signature
)

assert signature_valid is True

fresh_decision = provenance_decision(
    signature_valid,
    source_hash == source_hash
)

modified_decision = provenance_decision(
    signature_valid,
    source_hash == modified_hash
)

different_decision = provenance_decision(
    signature_valid,
    source_hash == different_hash
)

print("Test A: Fresh embedded audio")
print("  SOURCE MATCH")
print("  PROVENANCE VERIFIED")
print("  FINAL:", fresh_decision)

print("\nTest B: Modified audio")
print("  SOURCE DIFFERENT FROM VERIFIED FILE")
print("  FINAL:", modified_decision)

print("\nTest C: Different audio file")
print("  SOURCE DIFFERENT FROM VERIFIED FILE")
print("  FINAL:", different_decision)
