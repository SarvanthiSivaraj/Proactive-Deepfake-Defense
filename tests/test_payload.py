import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from src.payload.metadata import (
    generate_metadata
)

from src.payload.serialize import (
    serialize_payload
)

metadata = generate_metadata()

print(

    "Metadata:"
)

print(metadata)

serialized = serialize_payload(

    metadata
)

print()

print(

    "Serialized Bytes:"
)

print(serialized)

print()

print(

    "Length:",

    len(serialized)
)