from datetime import datetime, timezone
import hashlib

import numpy as np


DEFAULT_AUDIO_ID = "AUDIO001"
DEFAULT_GENERATOR = "VOICE_GEN_V1"
DEFAULT_CREATOR = "Sarvanthikha"
DEFAULT_ORGANIZATION = "Amrita University"
DEFAULT_MODEL_VERSION = "ProactiveDefense-v1.0"


def compute_audio_hash(audio, sample_rate=None):

    hasher = hashlib.sha256()

    normalized_audio = np.ascontiguousarray(
        np.asarray(
            audio,
            dtype=np.float32
        )
    )

    hasher.update(
        normalized_audio.tobytes()
    )

    if sample_rate is not None:

        hasher.update(
            str(sample_rate).encode()
        )

    return hasher.hexdigest()


def generate_metadata(
    source_hash="",
        timestamp=None,
        audio_id=DEFAULT_AUDIO_ID,
        generator=DEFAULT_GENERATOR,
        creator=DEFAULT_CREATOR,
        organization=DEFAULT_ORGANIZATION,
    model_version=DEFAULT_MODEL_VERSION
):

    if timestamp is None:

        timestamp = datetime.now(
            timezone.utc
        ).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    metadata={

        "id":audio_id,

        "generator":generator,

        "timestamp":timestamp,

        "creator":creator,

        "organization":organization,

        "model_version":model_version,

        "source_hash":source_hash
    }

    return metadata
