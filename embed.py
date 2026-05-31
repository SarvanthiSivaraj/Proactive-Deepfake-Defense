import os
import shutil
import zlib
import numpy as np

from reedsolo import RSCodec

from src.preprocessing.loader import *
from src.preprocessing.stft import *

from src.encoder.qim import *
from src.decoder.qim_decoder import *

from src.payload.serialize import *
from src.payload.bitstream import *

from src.security.signature import *


INPUT_DIR="input_audio"
OUTPUT_DIR="output"

RS_PARITY=160
SEED=42


def embed_audio(filename):

    filepath=os.path.join(
        INPUT_DIR,
        filename
    )

    if not os.path.exists(filepath):

        print(
            "\nFile not found:",
            filepath
        )

        return

    print(
        "\nEMBEDDING WATERMARK"
    )

    print(
        "-------------------"
    )

    # ---------- load ----------

    audio,sr=load_audio(
        filepath
    )

    stft=compute_stft(
        audio
    )

    magnitude,phase=split_mag_phase(
        stft
    )

    print(
        "STFT shape:",
        magnitude.shape
    )

    # ---------- deterministic metadata ----------

    metadata={

        "id":"AUDIO001",

        "generator":"VOICE_GEN_V1"
    }

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

    rsc=RSCodec(
        RS_PARITY
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

    # ---------- embed ----------

    embedded_mag,groups=embed_payload_qim(

        magnitude,

        payload_bits,

        seed=SEED
    )

    # ---------- reconstruct ----------

    watermarked=inverse_stft(

        merge_mag_phase(

            embedded_mag,

            phase
        )
    )

    # ---------- direct RAM debug ----------

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

    recovered_bits=recovered_bits[
        :len(payload_bits)
    ]

    direct_ber=np.mean(

        np.array(payload_bits)

        !=

        np.array(recovered_bits)
    )

    print(
        "\nDIRECT RAM BER:"
    )

    print(
        round(
            float(direct_ber),
            4
        )
    )

    # ---------- remainder debug ----------

    print(
        "\nREMAINDER DEBUG"
    )

    for bit,group in zip(

            payload_bits[:20],

            groups[:20]
    ):

        row,col=group[0]

        value=wm_mag[
            row,
            col
        ]

        remainder=np.mod(
            value,
            0.5
        )

        print(

            bit,

            " remainder=",

            round(
                float(remainder),
                4
            )
        )

    # ---------- output dirs ----------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        INPUT_DIR,
        exist_ok=True
    )

    output_path=os.path.join(

        OUTPUT_DIR,

        "protected.wav"
    )

    save_audio(

        output_path,

        watermarked,

        sr
    )

    verify_ready_path=os.path.join(

        INPUT_DIR,

        "protected.wav"
    )

    shutil.copyfile(

        output_path,

        verify_ready_path
    )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )

    print(
        verify_ready_path
    )

    print(
        "\nEMBED SUCCESS"
    )


if __name__=="__main__":

    filename=input(

        "\nEnter audio filename: "
    )

    embed_audio(
        filename
    )