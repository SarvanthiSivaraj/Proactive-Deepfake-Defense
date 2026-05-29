from reedsolo import RSCodec


def rs_encode(

        payload,

        parity_bytes=16
):

    rsc = RSCodec(

        parity_bytes
    )

    encoded = rsc.encode(

        payload
    )

    return encoded