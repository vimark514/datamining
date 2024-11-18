import numpy as np
import math
import cv2


def shannon_entropy(image_channel):
    histogram, _ = np.histogram(image_channel, bins=256, range=(0, 256), density=True)
    histogram = histogram[histogram > 0]
    entropy = -np.sum(histogram * np.log2(histogram))
    return entropy


def hartley_measure(image_channel):
    unique_values = np.unique(image_channel)
    hartley = math.log2(len(unique_values))
    return hartley


def markov_process_first_order(image_channel):
    transition_matrix = np.zeros((256, 256), dtype=np.float64)
    h, w = image_channel.shape

    for i in range(h):
        for j in range(w):
            current_pixel = image_channel[i, j]
            if j < w - 1:
                next_pixel_right = image_channel[i, j + 1]
                transition_matrix[current_pixel, next_pixel_right] += 1
            if i < h - 1:
                next_pixel_down = image_channel[i + 1, j]
                transition_matrix[current_pixel, next_pixel_down] += 1

    total_transitions = np.sum(transition_matrix)
    if total_transitions == 0:
        return 0.0

    P_x = np.sum(transition_matrix, axis=1) / total_transitions

    with np.errstate(divide='ignore', invalid='ignore'):
        transition_matrix = np.divide(transition_matrix, np.sum(transition_matrix, axis=1, keepdims=True),
                                      where=np.sum(transition_matrix, axis=1, keepdims=True) != 0)

    entropy_markov = 0.0
    for x in range(256):
        if P_x[x] > 0:
            P_y_given_x = transition_matrix[x, :]
            P_y_given_x_nonzero = P_y_given_x[P_y_given_x > 0]
            entropy_x = -np.sum(P_y_given_x_nonzero * np.log2(P_y_given_x_nonzero))
            entropy_markov += P_x[x] * entropy_x

    return entropy_markov


def add_gaussian_noise(image, mean=0, var=0.01):
    row, col, ch = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def count_noisy_pixels(original, noisy):
    diff = cv2.absdiff(original, noisy)
    noisy_pixels = np.count_nonzero(diff)
    return noisy_pixels


def calculate_normalized_correlation(original, noisy):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float64)
    noisy_gray = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY).astype(np.float64)

    if original_gray.shape != noisy_gray.shape:
        raise ValueError("Images must be of the same size.")

    mean_original = np.mean(original_gray)
    mean_noisy = np.mean(noisy_gray)

    diff_original = original_gray - mean_original
    diff_noisy = noisy_gray - mean_noisy

    numerator = np.sum(diff_original * diff_noisy)

    denominator = np.sqrt(np.sum(diff_original ** 2) * np.sum(diff_noisy ** 2))

    if denominator == 0:
        return 0

    normalized_correlation = numerator / denominator

    return normalized_correlation


def mean_squared_error(original, noisy):
    return np.mean((original - noisy) ** 2)


def calculate_psnr(original, noisy):
    mse = mean_squared_error(original, noisy)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return np.inf

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def calculate_max_error(original, noisy):
    error = np.abs(original.astype(np.float32) - noisy.astype(np.float32))
    return np.max(error)
