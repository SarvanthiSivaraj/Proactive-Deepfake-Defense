import sys
import os

sys.path.append(
    os.path.abspath(".")
)

import numpy as np

from src.sync.gold import *
from src.sync.synchronizer import *


sync = generate_gold_code()

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


result = find_sync(

    signal,

    sync
)

print("\nSync Result:")

print(result)