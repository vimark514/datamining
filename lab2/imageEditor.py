import numpy as np


def rgb_to_grayscale(rgb_image):
    h, w, _ = rgb_image.shape
    gray_image = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            r, g, b = rgb_image[y, x]
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            gray_image[y, x] = gray_value

    return gray_image


def threshold_processing(g_image, threshold):
    thresholded = np.zeros_like(g_image)
    thresholded[g_image >= threshold] = 255
    thresholded[g_image < threshold] = 0
    return thresholded


def segment_image(image, block_size):
    h, w = image.shape
    segments = []
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            segment = image[y:y + block_size, x:x + block_size]
            segments.append(segment)
    return segments


def segment_rgb_image(image, block_size):
    h, w, _ = image.shape
    segments = []

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            segment = image[y:min(y + block_size, h), x:min(x + block_size, w)]
            segments.append(segment)

    return segments
