import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from src.ecc.encode import *
from src.ecc.decode import *


payload = b"HELLO WATERMARK"

print(

    "\nOriginal:"
)

print(payload)


encoded = rs_encode(

    payload
)

print(

    "\nEncoded:"
)

print(encoded)


corrupted = bytearray(

    encoded
)

corrupted[3] = 99
corrupted[8] = 55
corrupted[10] = 88

print(

    "\nCorrupted:"
)

print(corrupted)


decoded = rs_decode(

    corrupted
)

print(

    "\nRecovered:"
)

print(decoded)