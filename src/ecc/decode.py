def rs_decode(

        encoded_payload,

        parity_bytes=16
):

    try:
        from reedsolo import RSCodec, ReedSolomonError
    except ImportError as exc:
        raise RuntimeError(
            "The 'reedsolo' package is required for ECC decoding. Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc

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