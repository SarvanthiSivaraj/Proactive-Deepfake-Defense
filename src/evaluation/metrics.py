def compute_ber(

        original,

        recovered
):

    errors = 0

    for o, r in zip(

            original,

            recovered
    ):

        if o != r:

            errors += 1

    ber = (

        errors /

        len(original)
    )

    return ber