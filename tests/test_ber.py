import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from src.evaluation.metrics import *

original = [

    1,0,1,1,

    0,0,1,1
]

recovered = [

    1,1,0,0,

    1,1,1,0
]

ber = compute_ber(

    original,

    recovered
)

print(

    "\nBER:",

    ber
)