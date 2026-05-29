import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from src.encoder.qim import *
from src.decoder.qim_decoder import *


original = 3.42

bit = 0

embedded = embed_qim(

    original,

    bit
)

print(

    "\nOriginal:",

    original
)

print(

    "Embedded:",

    embedded
)

recovered = extract_qim(

    embedded
)

print(

    "Recovered Bit:",

    recovered
)