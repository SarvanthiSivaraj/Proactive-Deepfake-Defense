import sys
import os
import numpy as np

sys.path.append(
    os.path.abspath(".")
)

from src.preprocessing.loader import *
from src.preprocessing.stft import *

from src.encoder.qim import *
from src.payload.bitstream import *

audio,sr=load_audio(
    "data/eval_audio/speech2.wav"
)

stft=compute_stft(
    audio
)

mag,phase=split_mag_phase(
    stft
)

payload=[0,1]*200

embedded_mag,groups=embed_payload_qim(

    mag,

    payload,

    seed=42
)

remainders=[]

labels=[]

for bit,group in zip(

        payload,

        groups
):

    for row,col in group:

        rem=np.mod(

            embedded_mag[
                row,
                col
            ],

            0.5
        )

        remainders.append(
            rem
        )

        labels.append(
            bit
        )

remainders=np.array(
    remainders
)

labels=np.array(
    labels
)

zero_cluster=remainders[
    labels==0
]

one_cluster=remainders[
    labels==1
]

print(
    "\n0-BIT CLUSTER"
)

print(
    "mean:",
    np.mean(
        zero_cluster
    )
)

print(
    "std:",
    np.std(
        zero_cluster
    )
)

print(
    "\n1-BIT CLUSTER"
)

print(
    "mean:",
    np.mean(
        one_cluster
    )
)

print(
    "std:",
    np.std(
        one_cluster
    )
)