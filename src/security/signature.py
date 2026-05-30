import hashlib


def sign_payload(
        payload_bytes
):

    digest = hashlib.sha256(

        payload_bytes

    ).hexdigest()

    return digest


def verify_signature(

        payload_bytes,

        signature
):

    computed = hashlib.sha256(

        payload_bytes

    ).hexdigest()

    return (

        computed==signature
    )