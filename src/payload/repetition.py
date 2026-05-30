def repeat_bits(
        bits,
        k=3
):

    repeated=[]

    for b in bits:

        repeated.extend(
            [b]*k
        )

    return repeated


def majority_decode(
        bits,
        k=3
):

    decoded=[]

    for i in range(
        0,
        len(bits),
        k
    ):

        chunk=bits[
            i:i+k
        ]

        ones=sum(
            chunk
        )

        if ones >= (k//2)+1:

            decoded.append(
                1
            )

        else:

            decoded.append(
                0
            )

    return decoded