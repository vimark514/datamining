import cv2
import matplotlib.pyplot as plt
import numpy as np
from average_diff import AverageDifferenceCalculator
from heatmap_plotter import Heatmap2DPlotter
from normalized_correlation import NormalizedCorrelation
from segmentator import Segmentator
from shennon_entropy import *

IMAGE_PATH = '../photos/I20.BMP'
SEGMENT_SIZE = 16

ENTROPY_THRESHOLD = 3.5
DIFFERENCE_THRESHOLD = 10
CORRELATION_THRESHOLD = 0.5


def get_image_from(path):
    return cv2.imread(path)


def get_graystale_image_of(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_shennon_entropies_for_each_segment(segments):
    return [*map(lambda s: ShennonEntropyCalculator(s).calculate(), segments)]


def get_average_difference_for_each_segment(segments):
    return [*map(lambda s: AverageDifferenceCalculator(s).calculate(), segments)]


def get_normalized_correlation_for_each_segment(segments):
    return [*map(lambda s: NormalizedCorrelation(s).calculate(), segments)]


def make_2d_array_from_1d_array(segments, segment_size, original_image):
    h, w = original_image.shape
    num_segments_width = w // segment_size
    num_segments_height = len(segments) // num_segments_width
    return np.reshape(segments, (num_segments_height, num_segments_width))


def show_2d_heatmap_for(data, title=""):
    data_2d = make_2d_array_from_1d_array(data, SEGMENT_SIZE, image)
    Heatmap2DPlotter(data_2d, title=title).show()


def plot_histogram(data, title):
    plt.figure()
    plt.hist(data, bins=50, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()


def classify_segments_by_threshold(data, threshold):
    return [1 if value > threshold else 0 for value in data]


def visualize_classification(classified_data, original_image, title):
    classified_2d = make_2d_array_from_1d_array(classified_data, SEGMENT_SIZE, original_image)
    plt.imshow(classified_2d, cmap='gray', interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    image = get_image_from(IMAGE_PATH)
    image = get_graystale_image_of(image)

    if image is None:
        raise FileNotFoundError(f"Image not found: '{IMAGE_PATH}'")

    segments = Segmentator(image, SEGMENT_SIZE).segment_image()

    shennon_entropies = get_shennon_entropies_for_each_segment(segments)
    show_2d_heatmap_for(shennon_entropies,
                        f"Classification by Shannon entropy (segments {SEGMENT_SIZE}x{SEGMENT_SIZE})")
    plot_histogram(shennon_entropies, "Shannon entropy histogram")
    classified_entropy = classify_segments_by_threshold(shennon_entropies, ENTROPY_THRESHOLD)
    visualize_classification(classified_entropy, image, "Classification by Shannon entropy")

    average_differences = get_average_difference_for_each_segment(segments)
    average_differences = np.array(average_differences).flatten()

    show_2d_heatmap_for(average_differences,
                        f"Classification by mean square deviation (segments {SEGMENT_SIZE}x{SEGMENT_SIZE})")
    plot_histogram(average_differences, "Mean square difference histogram")
    classified_differences = classify_segments_by_threshold(average_differences, DIFFERENCE_THRESHOLD)
    visualize_classification(classified_differences, image, "Classification by mean square difference")

    normalized_correlations = get_normalized_correlation_for_each_segment(segments)
    show_2d_heatmap_for(normalized_correlations,
                        f"Classification by normalised correlation coefficient (segments {SEGMENT_SIZE}x{SEGMENT_SIZE})")
    plot_histogram(normalized_correlations, "Normalized correlation histogram")
    classified_correlations = classify_segments_by_threshold(normalized_correlations, CORRELATION_THRESHOLD)
    visualize_classification(classified_correlations, image, "Classification by normalised correlation coefficient")
