import sys
import os
import numpy as np

sys.path.append(
    os.path.abspath(".")
)

from src.encoder.qim import *
from src.decoder.qim_decoder import *

np.random.seed(
    42
)

mag=np.random.rand(
    100,
    100
)*5

payload=[0,1]*200

embedded,groups=embed_payload_qim(

    mag,

    payload,

    seed=42
)

recovered=extract_payload_qim(

    embedded,

    groups
)

ber=np.mean(

    np.array(
        payload
    )

    !=

    np.array(
        recovered
    )
)

print(
    "\nUNIT TEST BER:"
)

print(
    ber
)