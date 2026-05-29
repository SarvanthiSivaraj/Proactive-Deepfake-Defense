import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from src.payload.metadata import *
from src.payload.serialize import *

from src.crypto.signer import *

from src.ecc.encode import *
from src.ecc.decode import *


# STEP 1

metadata = generate_metadata()

print("\nMetadata:")

print(metadata)


# STEP 2

payload = serialize_payload(

    metadata
)

print("\nSerialized Payload:")

print(payload)


# STEP 3

private_key, public_key = generate_keys()

signature = sign_message(

    private_key,

    payload
)

print(

    "\nSignature Length:",

    len(signature)
)


# STEP 4

protected_signature = rs_encode(

    signature
)

print(

    "\nECC Protected Signature Length:",

    len(protected_signature)
)


# STEP 5

corrupted = bytearray(

    protected_signature
)

corrupted[5] = 99
corrupted[9] = 42
corrupted[15] = 77


# STEP 6

recovered_signature = rs_decode(

    corrupted
)


# STEP 7

verified = verify_signature(

    public_key,

    recovered_signature,

    payload
)

print(

    "\nVerification Result:"
)

print(

    verified
)