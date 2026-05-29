import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from src.crypto.signer import *
from src.payload.metadata import *
from src.payload.serialize import *


# ORIGINAL METADATA

metadata = generate_metadata()

original_payload = serialize_payload(
    metadata
)

private_key, public_key = generate_keys()

signature = sign_message(
    private_key,
    original_payload
)


# TAMPER AFTER SIGNING

metadata["generator"] = "FAKE_MODEL"

tampered_payload = serialize_payload(
    metadata
)


verified = verify_signature(
    public_key,
    signature,
    tampered_payload
)

print("\nVerification:")

print(verified)