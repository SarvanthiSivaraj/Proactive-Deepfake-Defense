import subprocess
import sys

thresholds = [

    0.12,
    0.13,
    0.14,
    0.15,
    0.16
]

print(
    "\nLOWPASS DECODER SWEEP"
)

for t in thresholds:

    print(
        "\n==================="
    )

    print(
        "THRESHOLD =",
        t
    )

    print(
        "==================="
    )

    with open(

        "src/decoder/qim_decoder.py",

        "w"

    ) as f:

        f.write(f'''
import numpy as np

DELTA = 0.5
THRESHOLD = {t}


def extract_qim(value):

    remainder=np.mod(
        value,
        DELTA
    )

    if remainder < THRESHOLD:

        return 0

    else:

        return 1


def extract_payload_qim(

        magnitude,

        grouped_locations
):

    bits=[]

    for group in grouped_locations:

        votes=[]

        for row,col in group:

            votes.append(

                extract_qim(

                    magnitude[
                        row,
                        col
                    ]
                )
            )

        bits.append(

            int(
                np.mean(
                    votes
                )>=0.5
            )
        )

    return bits
''')

    subprocess.run(

        [

            sys.executable,

            "tests/test_extended_attack_suite.py"

        ]

    )