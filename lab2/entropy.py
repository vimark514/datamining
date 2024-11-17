import numpy as np
import math


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
