import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from lab5.amount_of_big_diff_calculator import AmountOfBigDifferenceCalculator
from lab5.average_diff import AverageDifferenceCalculator
from lab5.heatmap_plotter import Heatmap2DPlotter
from lab5.segmentator import Segmentator
from lab5.max_diff_equal_series_calculator import MaxDifferenceEqualSeriesCalculator
from lab5.shennon_entropy import ShennonEntropyCalculator

MAX_COLOR_BRIGHTNESS = 255

mpl.rcParams['figure.dpi'] = 120
IMAGE_PATH = '../photos/I05.BMP'
SEGMENT_SIZE = 8
MAX_DIFF_TO_TREAT_AS_EQUAL = 20
MIN_DIFF_TO_TREAT_AS_DIFFERENT = MAX_COLOR_BRIGHTNESS - MAX_DIFF_TO_TREAT_AS_EQUAL


def get_image_from(path):
    return cv2.imread(path)


def get_grayscale_image_of(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def make_2d_array_from_1d_array(segments, segment_size, original_image):
    h, w = original_image.shape
    num_segments_width = w // segment_size
    num_segments_height = len(segments) // num_segments_width
    return np.reshape(segments, (num_segments_height, num_segments_width))


def show_2d_heatmap_for(data, title=""):
    data_2d = make_2d_array_from_1d_array(data, SEGMENT_SIZE, image)

    Heatmap2DPlotter(data_2d, title=title).show()


def get_equal_series_for_each_segment(segments):
    calculator = MaxDifferenceEqualSeriesCalculator(MAX_DIFF_TO_TREAT_AS_EQUAL)
    return [*map(lambda s: calculator.calculate(s), segments)]


def get_amount_of_big_diff_for_each_segment(segments):
    calculator = AmountOfBigDifferenceCalculator(MIN_DIFF_TO_TREAT_AS_DIFFERENT)
    return [*map(lambda s: calculator.calculate(s), segments)]


def get_shennon_entropies_for_each_segment(segments):
    return [*map(lambda s: ShennonEntropyCalculator(s).calculate(), segments)]


def get_average_difference_for_each_segment(segments):
    return [*map(lambda s: AverageDifferenceCalculator(s).calculate(), segments)]


if __name__ == "__main__":
    image = get_image_from(IMAGE_PATH)
    image = get_grayscale_image_of(image)

    if image is None:
        raise FileNotFoundError(f"Image not found: '{IMAGE_PATH}'")

    segments = Segmentator(image, SEGMENT_SIZE).segment_image()

    equal_series = get_equal_series_for_each_segment(segments)

    diff_series = get_amount_of_big_diff_for_each_segment(segments)

    shennon_entropies = get_shennon_entropies_for_each_segment(segments)
    average_differences = get_average_difference_for_each_segment(segments)

    show_2d_heatmap_for(equal_series,
                        f"Segment classification result\nby the length of identical elements\n(segments {SEGMENT_SIZE}x{SEGMENT_SIZE}, max difference: {MAX_DIFF_TO_TREAT_AS_EQUAL})")

    show_2d_heatmap_for(diff_series,
                        f"Segment classification result\nby the number of brightness changes\n(segments {SEGMENT_SIZE}x{SEGMENT_SIZE}, max difference: {MIN_DIFF_TO_TREAT_AS_DIFFERENT})")

    show_2d_heatmap_for(shennon_entropies,
                        f"Segment classification result\nby Shannon entropy\n(segments {SEGMENT_SIZE}x{SEGMENT_SIZE})")

    show_2d_heatmap_for(average_differences,
                        f"Segment classification result\nby mean square deviation\n(segments {SEGMENT_SIZE}x{SEGMENT_SIZE})")
