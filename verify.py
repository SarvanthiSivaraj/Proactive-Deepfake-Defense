import os
import pickle
import numpy as np
import zlib

from src.preprocessing.loader import *
from src.preprocessing.stft import *

from src.decoder.qim_decoder import *

from src.payload.bitstream import *
from src.payload.serialize import *
from src.payload.metadata import *

from src.security.signature import *
from src.forensic.classifier import classify_attack as classify_attack_rules
from src.forensic.ml import classify_attack_ml
from src.forensic.report import build_attack_report, decide_final_authentication

from src.ecc.decode import rs_decode


INPUT_DIR="input_audio"
OUTPUT_DIR="output"

RS_PARITY=160
ATTACK_MODE=os.environ.get("FORENSIC_ATTACK_MODE", "rules").strip().lower()


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

    verified_audio_hash=compute_audio_hash(
        audio,
        sr
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
    source_hash_match=False

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

    # ---------- provenance hash ----------

    report.append(
        "Provenance Hash Status\n"
    )

    report.append(
        "----------------------\n"
    )

    if packet is None or not sig_ok:

        report.append(
            "  Status: SKIPPED\n"
        )

        report.append(
            "  Detail: Provenance unavailable.\n\n"
        )

    else:

        metadata=packet[
            "metadata"
        ]

        source_hash=metadata.get(
            "source_hash"
        )

        source_hash_match=(
            source_hash==verified_audio_hash
        )

        report.append(
            f"  Source Hash: {source_hash}\n"
        )

        report.append(
            f"  Verified Audio Hash: {verified_audio_hash}\n"
        )

        if source_hash_match:

            report.append(
                "  Status: SOURCE MATCH\n"
            )

            report.append(
                "  Detail: Source and verified audio hashes match.\n\n"
            )

        else:

            report.append(
                "  Status: SOURCE DIFFERENT FROM VERIFIED FILE\n"
            )

            report.append(
                "  Detail: Source audio hash differs from verified audio hash.\n\n"
            )

    # ---------- attack analysis ----------

    if ATTACK_MODE == "ml":

        attack_analysis=classify_attack_ml(
            audio,
            sr,
            magnitude,
            ber,
            source_hash_match=source_hash_match,
            ecc_success=ecc_ok,
        )

    else:

        attack_analysis=classify_attack_rules(
            audio,
            sr,
            magnitude,
            ber,
            source_hash_match=source_hash_match,
            ecc_success=ecc_ok,
        )

    report.extend(
        build_attack_report(
            attack_analysis,
            source_hash_match,
        )
    )

    final_result=decide_final_authentication(
        sig_ok,
        source_hash_match,
        attack_analysis,
    )

    # ---------- final result ----------

    report.append(
        "Final Authentication\n"
    )

    report.append(
        "--------------------\n"
    )

    report.append(
        f"  {final_result}\n\n"
    )

    # ---------- metadata ----------

    report.append(
        "Recovered Metadata\n"
    )

    report.append(
        "------------------\n"
    )

    if packet is not None and sig_ok:

        metadata=packet[
            "metadata"
        ]

        report.append(
            f"  ID: {metadata.get('id')}\n"
        )

        report.append(
            f"  Generator: {metadata.get('generator')}\n"
        )

        report.append(
            f"  Timestamp: {metadata.get('timestamp')}\n"
        )

        report.append(
            f"  Creator: {metadata.get('creator')}\n"
        )

        report.append(
            f"  Organization: {metadata.get('organization')}\n"
        )

        report.append(
            f"  Model Version: {metadata.get('model_version')}\n"
        )

        report.append(
            f"  Source Hash: {metadata.get('source_hash')}\n"
        )

        report.append(
            "\nPROVENANCE SUMMARY\n"
        )

        report.append(
            "-------------------\n"
        )

        report.append(
            f"Creator:\n{metadata.get('creator')}\n\n"
        )

        report.append(
            f"Organization:\n{metadata.get('organization')}\n\n"
        )

        report.append(
            f"Generated By:\n{metadata.get('generator')}\n\n"
        )

        report.append(
            f"Model Version:\n{metadata.get('model_version')}\n\n"
        )

        report.append(
            f"Timestamp:\n{metadata.get('timestamp')}\n"
        )

        report.append(
            "\nPROVENANCE MODEL\n"
        )

        report.append(
            "----------------\n"
        )

        report.append(
            "Source hash stored in metadata.\n"
        )

        report.append(
            "Verified audio hash computed from the file under inspection.\n"
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
