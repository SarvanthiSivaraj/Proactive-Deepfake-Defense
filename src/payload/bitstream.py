def bytes_to_bits(data):

    bits=[]

    for byte in data:

        for i in range(8):

            bits.append(

                (

                    byte >>

                    (7-i)

                ) & 1
            )

    return bits


def bits_to_bytes(bits):

    result=[]

    for i in range(

            0,

            len(bits),

            8
    ):

        byte=0

        chunk = bits[

            i:i+8
        ]

        for bit in chunk:

            byte=(

                byte<<1

            ) | bit

        result.append(

            byte
        )

    return bytes(
        result
    )