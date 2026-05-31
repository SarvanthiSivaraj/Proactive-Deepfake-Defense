def interleave_bits(bits, rows=8):

    cols = (

        len(bits)
        +
        rows
        -
        1

    ) // rows

    matrix = []

    idx = 0

    for r in range(rows):

        row = []

        for c in range(cols):

            if idx < len(bits):

                row.append(
                    bits[idx]
                )

            else:

                row.append(
                    0
                )

            idx += 1

        matrix.append(
            row
        )

    output = []

    for c in range(cols):

        for r in range(rows):

            output.append(

                matrix[r][c]

            )

    return output


def deinterleave_bits(bits, rows=8):

    cols = (

        len(bits)
        +
        rows
        -
        1

    ) // rows

    matrix = [

        [0]*cols

        for _ in range(rows)
    ]

    idx = 0

    for c in range(cols):

        for r in range(rows):

            if idx < len(bits):

                matrix[r][c] = bits[idx]

            idx += 1

    output = []

    for r in range(rows):

        for c in range(cols):

            output.append(

                matrix[r][c]

            )

    return output[
        :len(bits)
    ]