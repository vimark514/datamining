import numpy as np
from lab5.matrix_creator import MatrixCreator


class MaxDifferenceEqualSeriesCalculator:
    def __init__(self, max_diff):
        self.max_diff = max_diff

    def calculate(self, image):
        image = image.astype(np.int16)

        self.__init_values()

        image_as_matrix = MatrixCreator.from_vector(image)

        for matrix in image_as_matrix:
            for value in matrix:
                diff_between_prev_and_cur = abs(value - self.prev_value)
                is_small_difference = diff_between_prev_and_cur <= self.max_diff

                if is_small_difference:
                    self.__increment_cur_max_value()

                self.__save_prev_value(value)

            self.__save_current_max_for_a_row()

        avg_max_value = sum(self.max_values_per_row) / len(self.max_values_per_row)
        return avg_max_value

    def __init_values(self):
        self.max_values_per_row = []
        self.prev_value = 0
        self.cur_max_value = 0

    def __save_prev_value(self, value):
        self.prev_value = value

    def __save_current_max_for_a_row(self):
        self.max_values_per_row.append(self.cur_max_value)
        self.cur_max_value = 0

    def __increment_cur_max_value(self):
        self.cur_max_value += 1
