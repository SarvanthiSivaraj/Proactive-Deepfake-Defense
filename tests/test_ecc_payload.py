import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from src.ecc.encode import *
from src.ecc.decode import *

from src.payload.bitstream import *


payload = b"HELLO"


print("\nOriginal:")

print(payload)


encoded = rs_encode(

    payload
)

print("\nECC Encoded:")

print(encoded)


bits = bytes_to_bits(

    encoded
)

print(

    "\nBit Length:"
)

print(

    len(bits)
)


reconstructed = bits_to_bytes(

    bits
)


decoded = rs_decode(

    reconstructed
)

print("\nRecovered:")

print(decoded)