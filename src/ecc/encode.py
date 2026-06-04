def rs_encode(

        payload,

        parity_bytes=16
):

    try:
        from reedsolo import RSCodec
    except ImportError as exc:
        raise RuntimeError(
            "The 'reedsolo' package is required for ECC encoding. Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    rsc = RSCodec(

        parity_bytes
    )

    encoded = rsc.encode(

        payload
    )

    return encoded