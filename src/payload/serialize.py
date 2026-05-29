import json


def serialize_payload(

        payload
):

    serialized = json.dumps(

        payload,

        sort_keys=True
    )

    return serialized.encode()