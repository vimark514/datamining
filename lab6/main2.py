import numpy as np
import matplotlib.pyplot as plt
import cv2
from lab5.segmentator import Segmentator
from lab5.max_diff_equal_series_calculator import MaxDifferenceEqualSeriesCalculator
from lab5.amount_of_big_diff_calculator import AmountOfBigDifferenceCalculator
from lab5.shennon_entropy import ShennonEntropyCalculator

IMAGE_PATH = '../photos/I05.BMP'
SEGMENT_SIZE = 16
MAX_DIFF_TO_TREAT_AS_EQUAL = 10
MIN_DIFF_TO_TREAT_AS_DIFFERENT = 255 - MAX_DIFF_TO_TREAT_AS_EQUAL


def calculate_metrics(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    segments = Segmentator(image, SEGMENT_SIZE).segment_image()

    entropies = [ShennonEntropyCalculator(s).calculate() for s in segments]
    equal_series = [MaxDifferenceEqualSeriesCalculator(MAX_DIFF_TO_TREAT_AS_EQUAL).calculate(s) for s in segments]
    big_diffs = [AmountOfBigDifferenceCalculator(MIN_DIFF_TO_TREAT_AS_DIFFERENT).calculate(s) for s in segments]

    return entropies, equal_series, big_diffs


def plot_histograms(entropies, equal_series, big_diffs):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(entropies, bins=20, color='blue', alpha=0.7)
    axes[0].set_title("Histogram of Shannon Entropies")
    axes[0].set_xlabel("Entropy")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(equal_series, bins=20, color='green', alpha=0.7)
    axes[1].set_title("Histogram of Equal Series Lengths")
    axes[1].set_xlabel("Series Length")
    axes[1].set_ylabel("Frequency")

    axes[2].hist(big_diffs, bins=20, color='red', alpha=0.7)
    axes[2].set_title("Histogram of Brightness Transitions")
    axes[2].set_xlabel("Number of Transitions")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


entropies, equal_series, big_diffs = calculate_metrics(IMAGE_PATH)
plot_histograms(entropies, equal_series, big_diffs)

LOW_ENTROPY_THRESHOLD = np.percentile(entropies, 33)
HIGH_ENTROPY_THRESHOLD = np.percentile(entropies, 66)

LOW_TRANSITIONS_THRESHOLD = np.percentile(big_diffs, 33)
HIGH_TRANSITIONS_THRESHOLD = np.percentile(big_diffs, 66)

LONG_SERIES_THRESHOLD = np.percentile(equal_series, 66)
SHORT_SERIES_THRESHOLD = np.percentile(equal_series, 33)

print("Recommended Thresholds:")
print(f"LOW_ENTROPY_THRESHOLD = {LOW_ENTROPY_THRESHOLD:.2f}")
print(f"HIGH_ENTROPY_THRESHOLD = {HIGH_ENTROPY_THRESHOLD:.2f}")
print(f"LOW_TRANSITIONS_THRESHOLD = {LOW_TRANSITIONS_THRESHOLD:.2f}")
print(f"HIGH_TRANSITIONS_THRESHOLD = {HIGH_TRANSITIONS_THRESHOLD:.2f}")
print(f"LONG_SERIES_THRESHOLD = {LONG_SERIES_THRESHOLD:.2f}")
print(f"SHORT_SERIES_THRESHOLD = {SHORT_SERIES_THRESHOLD:.2f}")
