import sys
import os
import zlib
import numpy as np

sys.path.append(
    os.path.abspath(".")
)

from reedsolo import RSCodec
from scipy.signal import butter,filtfilt,resample

from src.preprocessing.loader import *
from src.preprocessing.stft import *

from src.encoder.qim import *
from src.decoder.qim_decoder import *

from src.payload.metadata import *
from src.payload.serialize import *
from src.payload.bitstream import *

from src.security.signature import *
from src.evaluation.metrics import *

RS_PARITY=160
SEED=42

np.random.seed(
    SEED
)

rsc=RSCodec(
    RS_PARITY
)

metadata=generate_metadata()

metadata_bytes=serialize_payload(
    metadata
)

signature=sign_payload(
    metadata_bytes
)

packet={

    "metadata":metadata,

    "signature":signature
}

packet_bytes=serialize_payload(
    packet
)

compressed=zlib.compress(
    packet_bytes
)

ecc_bytes=rsc.encode(
    compressed
)

payload_bits=bytes_to_bits(
    ecc_bytes
)

print(
    "\nPayload Bits:"
)

print(
    len(payload_bits)
)

audio,sr=load_audio(
    "data/sample_audio/speech.wav"
)

stft=compute_stft(
    audio
)

mag,phase=split_mag_phase(
    stft
)

embedded_mag,groups=embed_payload_qim(

    mag,

    payload_bits,

    seed=SEED
)

watermarked=inverse_stft(

    merge_mag_phase(

        embedded_mag,

        phase
    )
)

def lowpass_attack(x):

    b,a=butter(

        4,

        0.35,

        btype="low"
    )

    return filtfilt(
        b,
        a,
        x
    )


def resample_attack(x):

    down=resample(
        x,
        len(x)//2
    )

    up=resample(
        down,
        len(x)
    )

    return up


def crop_attack(x):

    cropped=x[
        1000:
    ]

    padded=np.pad(

        cropped,

        (0,1000)
    )

    return padded


attacks={

    "NONE":watermarked,

    "GAUSSIAN":

        watermarked
        +
        np.random.normal(

            0,
            0.0003,
            len(watermarked)
        ),

    "AMPLITUDE":

        watermarked*0.99,

    "LOWPASS":

        lowpass_attack(
            watermarked
        ),

    "RESAMPLE":

        resample_attack(
            watermarked
        ),

    "CROP":

        crop_attack(
            watermarked
        )
}

print(
    "\nEXTENDED ATTACK SUITE"
)

for attack_name,attacked_audio in attacks.items():

    print(
        "\n==================="
    )

    print(
        attack_name
    )

    print(
        "==================="
    )

    try:

        wm_stft=compute_stft(
            attacked_audio
        )

        wm_mag,_=split_mag_phase(
            wm_stft
        )

        if attack_name=="CROP":

            corrected=[]

            for group in groups:

                new_group=[]

                for row,col in group:

                    new_group.append(
                        (
                            row,
                            col-2
                        )
                    )

                corrected.append(
                    new_group
                )

            recovered_bits=extract_payload_qim(

                wm_mag,

                corrected
            )

        else:

            recovered_bits=extract_payload_qim(

                wm_mag,

                groups
            )

        ber=compute_ber(

            payload_bits,

            recovered_bits
        )

        print(
            "\nBER:"
        )

        print(
            ber
        )

        recovered_bytes=bits_to_bytes(
            recovered_bits
        )

        decoded=rsc.decode(

            recovered_bytes

        )[0]

        decompressed=zlib.decompress(
            decoded
        )

        recovered_packet=deserialize_payload(
            decompressed
        )

        verified=verify_signature(

            serialize_payload(

                recovered_packet[
                    "metadata"
                ]

            ),

            recovered_packet[
                "signature"
            ]
        )

        print(
            "\nECC SUCCESS"
        )

        print(
            "Signature:",
            verified
        )

    except Exception as e:

        print(
            "\nPIPELINE FAILED"
        )

        print(e)