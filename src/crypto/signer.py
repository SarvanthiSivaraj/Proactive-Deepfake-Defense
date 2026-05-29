from cryptography.hazmat.primitives.asymmetric.ed25519 import (

    Ed25519PrivateKey,

    Ed25519PublicKey
)

from cryptography.hazmat.primitives import serialization


def generate_keys():

    private_key = Ed25519PrivateKey.generate()

    public_key = private_key.public_key()

    return private_key, public_key


def save_keys(

        private_key,

        public_key
):

    with open(

        "keys/private_key.pem",

        "wb"

    ) as f:

        f.write(

            private_key.private_bytes(

                encoding=

                serialization.Encoding.PEM,

                format=

                serialization.PrivateFormat.PKCS8,

                encryption_algorithm=

                serialization.NoEncryption()
            )
        )

    with open(

        "keys/public_key.pem",

        "wb"

    ) as f:

        f.write(

            public_key.public_bytes(

                encoding=

                serialization.Encoding.PEM,

                format=

                serialization.PublicFormat.SubjectPublicKeyInfo
            )
        )


def sign_message(

        private_key,

        message
):

    return private_key.sign(

        message
    )
def verify_signature(

        public_key,

        signature,

        message
):

    try:

        public_key.verify(

            signature,

            message
        )

        return True

    except Exception:

        return False