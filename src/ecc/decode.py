from reedsolo import RSCodec, ReedSolomonError


def rs_decode(

        encoded_payload,

        parity_bytes=16
):

    rsc = RSCodec(

        parity_bytes
    )

    try:
        decoded = rsc.decode(

            encoded_payload
        )

        return decoded[0]
    except ReedSolomonError:
        return None