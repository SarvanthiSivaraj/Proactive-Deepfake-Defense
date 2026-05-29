import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from src.crypto.signer import *

message = b"Hello Watermark"

private_key, public_key = generate_keys()

print("Keys generated.")

save_keys(

    private_key,

    public_key
)

print("Keys saved.")

signature = sign_message(

    private_key,

    message
)

print(

    "Signature Length:",

    len(signature)
)