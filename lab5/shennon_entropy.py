import numpy as np


class ShennonEntropyCalculator:
    def __init__(self, image):
        self.image = image

    def calculate(self):
        histogram = self.__get_histogram()
        histogram = self.__remove_negative_histogram_values_from(histogram)
        return self.__get_entropy_from(histogram)

    def __get_histogram(self):
        histogram, _ = np.histogram(self.image, bins=256, range=(0, 256), density=True)
        return histogram

    def __remove_negative_histogram_values_from(self, histogram):
        return histogram[histogram > 0]

    def __get_entropy_from(self, histogram):
        return -np.sum(histogram * np.log2(histogram))
