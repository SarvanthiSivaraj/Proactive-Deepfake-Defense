import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from src.crypto.signer import *

original_message = b"Hello Watermark"

private_key, public_key = generate_keys()

signature = sign_message(

    private_key,

    original_message
)

tampered_message = b"Fake Message"

result = verify_signature(

    public_key,

    signature,

    tampered_message
)

print(

    "Verification:",

    result
)