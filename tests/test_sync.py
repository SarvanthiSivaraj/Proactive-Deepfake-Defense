import sys
import os

sys.path.append(
    os.path.abspath(".")
)

import numpy as np

from src.sync.gold import *
from src.sync.synchronizer import *


sync = generate_gold_code()

print(

    "\nGold Code:"
)

print(sync)


signal = np.random.choice(

    [-1,1],

    size=200
)

insert_pos = 80

signal[

    insert_pos:

    insert_pos+

    len(sync)

] = sync


idx, conf = find_sync(

    signal,

    sync
)

print(

    "\nDetected Position:",

    idx
)

print(

    "Confidence:",

    conf
)