import time


def generate_metadata():

    metadata = {

        "id": "AUDIO001",

        "timestamp": int(

            time.time()
        ),

        "generator":

        "VOICE_GEN_V1"
    }

    return metadata