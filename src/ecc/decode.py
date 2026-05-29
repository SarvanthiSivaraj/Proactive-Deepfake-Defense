from reedsolo import RSCodec


def rs_decode(

        encoded_payload,

        parity_bytes=16
):

    rsc = RSCodec(

        parity_bytes
    )

    decoded = rsc.decode(

        encoded_payload
    )

    return decoded[0]