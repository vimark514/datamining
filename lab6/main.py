import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from lab5.amount_of_big_diff_calculator import AmountOfBigDifferenceCalculator
from lab5.average_diff import AverageDifferenceCalculator
from lab5.heatmap_plotter import Heatmap2DPlotter
from lab5.segmentator import Segmentator
from lab5.max_diff_equal_series_calculator import MaxDifferenceEqualSeriesCalculator
from lab5.shennon_entropy import ShennonEntropyCalculator

MAX_COLOR_BRIGHTNESS = 255
mpl.rcParams['figure.dpi'] = 120
IMAGE_PATH = '../photos/I05.BMP'
SEGMENT_SIZE = 16
MAX_DIFF_TO_TREAT_AS_EQUAL = 10
MIN_DIFF_TO_TREAT_AS_DIFFERENT = MAX_COLOR_BRIGHTNESS - MAX_DIFF_TO_TREAT_AS_EQUAL
LOW_ENTROPY_THRESHOLD = 4.5
HIGH_ENTROPY_THRESHOLD = 4.99
LONG_SERIES_THRESHOLD = 6.12
SHORT_SERIES_THRESHOLD = 4.5


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
    return [calculator.calculate(s) for s in segments]


def get_amount_of_big_diff_for_each_segment(segments):
    calculator = AmountOfBigDifferenceCalculator(MIN_DIFF_TO_TREAT_AS_DIFFERENT)
    return [calculator.calculate(s) for s in segments]


def get_shennon_entropies_for_each_segment(segments):
    return [ShennonEntropyCalculator(s).calculate() for s in segments]


def get_average_difference_for_each_segment(segments):
    return [AverageDifferenceCalculator(s).calculate() for s in segments]


def classify_entropy_segments(entropies):
    return [0 if e < LOW_ENTROPY_THRESHOLD else 1 if e < HIGH_ENTROPY_THRESHOLD else 2 for e in entropies]


def classify_structural_segments(series_lengths):
    return [0 if l > LONG_SERIES_THRESHOLD else 2 if l < SHORT_SERIES_THRESHOLD else 1 for l in series_lengths]


def calculate_classification_errors(predicted_classes, expert_classes):
    num_classes = 3
    errors = {i: {'false_positive': 0, 'false_negative': 0, 'predicted_count': 0} for i in range(num_classes)}

    for predicted, expert in zip(predicted_classes, expert_classes):
        errors[predicted]['predicted_count'] += 1
        if predicted == expert:
            continue
        errors[predicted]['false_positive'] += 1
        errors[expert]['false_negative'] += 1

    overall_errors = {'false_positive': 0, 'false_negative': 0}
    error_percentages = {}
    overall_predictions = sum(error['predicted_count'] for error in errors.values())

    for class_id, error_data in errors.items():
        false_positive = error_data['false_positive']
        false_negative = error_data['false_negative']
        predicted_count = error_data['predicted_count']
        false_positive_percent = (false_positive / predicted_count * 100) if predicted_count > 0 else 0
        false_negative_percent = (false_negative / overall_predictions * 100) if overall_predictions > 0 else 0

        error_percentages[class_id] = {
            'false_positive': false_positive,
            'false_negative': false_negative,
            'false_positive_percent': false_positive_percent,
            'false_negative_percent': false_negative_percent
        }
        overall_errors['false_positive'] += false_positive
        overall_errors['false_negative'] += false_negative

    overall_false_positive_percent = (
            overall_errors['false_positive'] / overall_predictions * 100) if overall_predictions > 0 else 0
    overall_false_negative_percent = (
            overall_errors['false_negative'] / overall_predictions * 100) if overall_predictions > 0 else 0

    print("Classification Error Analysis:")
    for class_id, percentages in error_percentages.items():
        print(
            f"Class {class_id}: False Positives: {percentages['false_positive']} ({percentages['false_positive_percent']:.2f}%), "
            f"False Negatives: {percentages['false_negative']} ({percentages['false_negative_percent']:.2f}%)")

    print("Overall Errors:")
    print(f"Overall False Positives: {overall_errors['false_positive']} ({overall_false_positive_percent:.2f}%)")
    print(f"Overall False Negatives: {overall_errors['false_negative']} ({overall_false_negative_percent:.2f}%)")

    return error_percentages, overall_errors


def show_classification_heatmap(classifications, title="Classifications"):
    data_2d = make_2d_array_from_1d_array(classifications, SEGMENT_SIZE, image)
    cmap = mcolors.ListedColormap(['green', 'yellow', 'red'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(data_2d, cmap=cmap, norm=norm)
    plt.colorbar(ticks=[0, 1, 2], label='Classes')
    plt.title(title)
    plt.show()


def introduce_random_errors(expert_classes, error_probability=0.1):
    erroneous_classes = expert_classes.copy()
    num_classes = 3

    for i in range(len(erroneous_classes)):
        if random.random() < error_probability:
            new_class = random.choice([c for c in range(num_classes) if c != erroneous_classes[i]])
            erroneous_classes[i] = new_class

    return erroneous_classes


if __name__ == "__main__":
    image = get_image_from(IMAGE_PATH)
    image = get_grayscale_image_of(image)

    if image is None:
        raise FileNotFoundError(f"Image not found: '{IMAGE_PATH}'")

    segments = Segmentator(image, SEGMENT_SIZE).segment_image()
    equal_series = get_equal_series_for_each_segment(segments)
    diff_series = get_amount_of_big_diff_for_each_segment(segments)
    shennon_entropies = get_shennon_entropies_for_each_segment(segments)

    entropy_classifications = classify_entropy_segments(shennon_entropies)
    show_classification_heatmap(entropy_classifications, title="Shannon entropy")
    structural_classifications = classify_structural_segments(equal_series)
    show_classification_heatmap(structural_classifications, title="Series classifications")
    expert_classifications_entropy = introduce_random_errors(entropy_classifications, error_probability=0.1)
    errors_entropy, overall_entropy_errors = calculate_classification_errors(entropy_classifications,
                                                                             expert_classifications_entropy)
    print("Entropy Classifications Error Analysis:")
    for class_id, error in errors_entropy.items():
        print(
            f"Class {class_id} - False Positives: {error['false_positive']}, False Negatives: {error['false_negative']}")
    print("\n\n\n")
    expert_classifications_structural = introduce_random_errors(structural_classifications, error_probability=0.1)
    errors_structural, overall_structural_errors = calculate_classification_errors(structural_classifications,
                                                                                   expert_classifications_structural)
    print("Structural Classifications Error Analysis:")
    for class_id, error in errors_structural.items():
        print(
            f"Class {class_id} - False Positives: {error['false_positive']}, False Negatives: {error['false_negative']}")
