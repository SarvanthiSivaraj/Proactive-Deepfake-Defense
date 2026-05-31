import os
import pickle
import numpy as np
import zlib

from src.preprocessing.loader import *
from src.preprocessing.stft import *

from src.decoder.qim_decoder import *

from src.payload.bitstream import *
from src.payload.serialize import *

from src.security.signature import *

from src.ecc.decode import rs_decode


INPUT_DIR="input_audio"
OUTPUT_DIR="output"

RS_PARITY=160


def verify_audio(filename):

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
        "\nVERIFYING AUDIO"
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

    # ---------- grouped locations ----------

    location_file=(
        "metadata/"
        "grouped_locations.pkl"
    )

    if not os.path.exists(
        location_file
    ):

        print(
            "\nERROR: grouped_locations.pkl missing."
        )

        return

    with open(
        location_file,
        "rb"
    ) as f:

        grouped_locations=pickle.load(
            f
        )

    # ---------- payload bits ----------

    payload_file=(
        "metadata/"
        "payload_bits.pkl"
    )

    if not os.path.exists(
        payload_file
    ):

        print(
            "\nERROR: payload_bits.pkl missing."
        )

        return

    with open(
        payload_file,
        "rb"
    ) as f:

        payload_bits=pickle.load(
            f
        )

    # ---------- extraction ----------

    recovered_bits=extract_payload_qim(

        magnitude,

        grouped_locations
    )

    recovered_bits=recovered_bits[
        :len(payload_bits)
    ]

    ber=float(

        np.mean(

            np.array(payload_bits)

            !=

            np.array(recovered_bits)
        )
    )

    # ---------- report ----------

    report=[]

    report.append(
        "AUTHENTICITY REPORT\n"
    )

    report.append(
        "=====================\n\n"
    )

    # ---------- BER ----------

    report.append(
        "Raw BER\n"
    )

    report.append(
        "-------\n"
    )

    report.append(
        f"  BER: {ber:.4f}\n\n"
    )

    # ---------- ECC ----------

    report.append(
        "ECC Status\n"
    )

    report.append(
        "----------\n"
    )

    recovered_bytes=bits_to_bytes(
        recovered_bits
    )

    decoded=rs_decode(

        recovered_bytes,

        parity_bytes=RS_PARITY
    )

    ecc_ok=False
    packet=None
    sig_ok=False

    if decoded is None:

        report.append(
            "  Status: FAILED\n"
        )

        report.append(
            "  Detail: Too many channel errors to correct.\n\n"
        )

    else:

        ecc_ok=True

        report.append(
            "  Status: SUCCESS\n\n"
        )

    # ---------- payload recovery ----------

    report.append(
        "Payload Recovery Status\n"
    )

    report.append(
        "-----------------------\n"
    )

    if not ecc_ok:

        report.append(
            "  Status: SKIPPED\n"
        )

        report.append(
            "  Detail: ECC failed.\n\n"
        )

    else:

        try:

            decompressed=zlib.decompress(
                decoded
            )

            packet=deserialize_payload(
                decompressed
            )

            report.append(
                "  Status: SUCCESS\n\n"
            )

        except Exception as e:

            report.append(
                "  Status: FAILED\n"
            )

            report.append(
                f"  Detail: {str(e)}\n\n"
            )

    # ---------- signature ----------

    report.append(
        "Signature Status\n"
    )

    report.append(
        "----------------\n"
    )

    if packet is None:

        report.append(
            "  Status: SKIPPED\n"
        )

        report.append(
            "  Detail: Payload unavailable.\n\n"
        )

    else:

        try:

            metadata=packet[
                "metadata"
            ]

            signature=packet[
                "signature"
            ]

            metadata_bytes=serialize_payload(
                metadata
            )

            sig_ok=verify_signature(

                metadata_bytes,

                signature
            )

            if sig_ok:

                report.append(
                    "  Status: VALID\n"
                )

                report.append(
                    "  Detail: Signature verified.\n\n"
                )

            else:

                report.append(
                    "  Status: INVALID\n"
                )

                report.append(
                    "  Detail: Signature mismatch.\n\n"
                )

        except Exception as e:

            report.append(
                "  Status: ERROR\n"
            )

            report.append(
                f"  Detail: {str(e)}\n\n"
            )

    # ---------- final auth ----------

    report.append(
        "Final Authentication\n"
    )

    report.append(
        "--------------------\n"
    )

    if sig_ok:

        report.append(
            "  AUTHENTIC AUDIO\n\n"
        )

    else:

        report.append(
            "  NOT AUTHENTIC\n\n"
        )

    # ---------- metadata ----------

    report.append(
        "Recovered Metadata\n"
    )

    report.append(
        "------------------\n"
    )

    if packet is not None and sig_ok:

        for k,v in packet[
            "metadata"
        ].items():

            report.append(
                f"  {k}: {v}\n"
            )

    else:

        report.append(
            "  Recovery failed.\n"
        )

        report.append(
            "  Metadata unavailable.\n"
        )

    report.append(
        "\n"
    )

    # ---------- save ----------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    report_path=os.path.join(

        OUTPUT_DIR,

        "verification_report.txt"
    )

    with open(
        report_path,
        "w"
    ) as f:

        f.writelines(
            report
        )

    print()

    for line in report:

        print(
            line,
            end=""
        )

    print(
        "\nReport saved:",
        report_path
    )


if __name__=="__main__":

    filename=input(
        "\nEnter audio filename: "
    )

    verify_audio(
        filename
    )