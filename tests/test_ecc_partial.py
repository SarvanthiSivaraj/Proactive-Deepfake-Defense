import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from src.payload.bitstream import *

from src.ecc.encode import *
from src.ecc.decode import *


payload = b"HELLO"

print("\nOriginal:")

print(payload)


encoded = rs_encode(

    payload
)

print("\nEncoded:")

print(encoded)

print(

    "\nEncoded Length:"
)

print(

    len(encoded)
)


bits = bytes_to_bits(

    encoded
)

print(

    "\nTotal Bits:"
)

print(

    len(bits)
)


# SAME THING YOU DID IN WATERMARK TEST

bits = bits[:80]


recovered_bytes = bits_to_bytes(

    bits
)

print(

    "\nRecovered Bytes:"
)

print(

    recovered_bytes
)


print(

    "\nTrying ECC Decode..."
)

try:

    decoded = rs_decode(

        recovered_bytes
    )

    print(

        "\nECC Success:"
    )

    print(

        decoded
    )

except Exception as e:

    print(

        "\nECC Decode Failed:"
    )

    print(

        e
    )