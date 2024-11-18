import numpy as np
from math import sqrt, ceil


class MatrixCreator:
    @classmethod
    def from_vector(cls, vector):
        vector = np.matrix.flatten(vector)

        if vector is None:
            raise RuntimeError("Non-empty vector is expected, but an empty one is provided")

        side = int(ceil(sqrt(len(vector))))

        can_array_be_a_square_matrix = side ** 2 == len(vector)
        if can_array_be_a_square_matrix is False:
            raise RuntimeError()

        return np.array(vector).reshape(side, side)


if __name__ == "__main__":
    vector = [1, 2, 3, 4]

    matrix = MatrixCreator.from_vector(vector)

    print(matrix)
