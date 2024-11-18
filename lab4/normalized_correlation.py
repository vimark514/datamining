import numpy as np


class NormalizedCorrelation:
    def __init__(self, image):
        self.image = image

    def calculate(self):
        mean = np.mean(self.image)
        stddev = np.std(self.image)

        if stddev <= 0:
            return 0

        normalized_image = (self.image - mean) / stddev
        normalized_coefficient = np.mean(normalized_image)
        return normalized_coefficient
