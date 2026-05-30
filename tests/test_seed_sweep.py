import os

seeds = [1,2,3,4,5]

for s in seeds:

    print("\n===================")
    print("SEED =",s)
    print("===================")

    os.system(

        f"python tests/test_ecc_watermark_recovery.py {s}"

    )