import sys
import os
import numpy as np

sys.path.append(
    os.path.abspath(".")
)

from reedsolo import RSCodec

from src.preprocessing.loader import *
from src.preprocessing.stft import *

from src.encoder.qim import *
from src.decoder.qim_decoder import *
import src.decoder.qim_decoder as qim_decoder

from src.payload.bitstream import *
from src.payload.interleave import *

from src.evaluation.metrics import *

RS_PARITY=16
SEED=42
DIAGNOSTIC_PAYLOAD=b"A"

FILE_CONFIGS={

    "speech1.wav":{

        "rows":4,

        "threshold":0.18
    },

    "speech2.wav":{

        "rows":8,

        "threshold":0.15
    },

    "speech3.wav":{

        "rows":8,

        "threshold":0.15
    },

    "speech4.wav":{

        "rows":8,

        "threshold":0.15
    },

    "speech5.wav":{

        "rows":2,

        "threshold":0.19
    }
}

np.random.seed(
    SEED
)

folder="data/eval_audio"

files=[

    f for f in os.listdir(folder)

    if f.endswith(".wav")
]

print(
    "\nDATASET EVALUATION"
)

results=[]
ecc_successes=0

for filename in files:

    print(
        "\n==================="
    )

    print(
        filename
    )

    print(
        "==================="
    )

    try:

        # ==========================
        # LOAD AUDIO
        # ==========================

        audio,sr=load_audio(

            os.path.join(
                folder,
                filename
            )
        )

        stft=compute_stft(
            audio
        )

        mag,phase=split_mag_phase(
            stft
        )

        rows,cols=mag.shape

        print(
            "STFT shape:",
            mag.shape
        )

        small_rsc=RSCodec(
            RS_PARITY
        )

        ecc_bytes=small_rsc.encode(
            DIAGNOSTIC_PAYLOAD
        )

        payload_bits=bytes_to_bits(
            ecc_bytes
        )

        config=FILE_CONFIGS[

            filename

        ]

        payload_bits=interleave_bits(
            payload_bits

            ,

            rows=config[

                "rows"

            ]
        )

        print(
            "Payload bits:",
            len(payload_bits)
        )

        print(
            "Config:",
            config
        )

        qim_decoder.THRESHOLD=config[

            "threshold"

        ]

        # ==========================
        # EMBED
        # ==========================

        embedded_mag,groups=embed_payload_qim(

            mag,

            payload_bits,

            seed=SEED
        )

        modified=merge_mag_phase(

            embedded_mag,

            phase
        )

        watermarked=inverse_stft(
            modified
        )

        # ==========================
        # EXTRACT
        # ==========================

        wm_stft=compute_stft(
            watermarked
        )

        wm_mag,_=split_mag_phase(
            wm_stft
        )

        recovered_bits=extract_payload_qim(

            wm_mag,

            groups
        )

        ber=compute_ber(

            payload_bits,

            recovered_bits
        )

        results.append(
            ber
        )

        print(
            "BER:",
            ber
        )

        recovered_bits=deinterleave_bits(
            recovered_bits

            ,

            rows=config[

                "rows"

            ]
        )

        recovered_bytes=bits_to_bytes(
            recovered_bits
        )

        try:

            decoded=small_rsc.decode(

                recovered_bytes

            )[0]

            print(
                "ECC SUCCESS"
            )

            ecc_successes+=1

        except Exception as ecc_error:

            print(
                "ECC FAILED"
            )

            print(
                ecc_error
            )

    except Exception as e:

        print(
            "PIPELINE FAILED"
        )

        print(e)

if len(results)>0:

    print(
        "\n==================="
    )

    print(
        "AVERAGE BER"
    )

    print(
        np.mean(
            results
        )
    )

    print(
        "ECC SUCCESS COUNT:"
    )

    print(
        ecc_successes,
        "/",
        len(results)
    )