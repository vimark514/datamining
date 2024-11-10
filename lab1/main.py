import cv2
import numpy as np
import matplotlib.pyplot as plt

file_path = '../photos/I05.BMP'

image = cv2.imread(file_path)

if image is None:
    raise FileNotFoundError(f"Image not found: '{file_path}'")
else:
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

    def segment_image(g_image, block_size):
        h, w = g_image.shape
        segments = []
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                segment = g_image[y:y + block_size, x:x + block_size]
                segments.append(segment)
        return segments


    gray_image = rgb_to_grayscale(image)
    plt.figure(figsize=(10, 7))
    plt.hist(gray_image.ravel(), bins=256, range=(0, 256), color='black')
    plt.title('Histogram of gray image')
    plt.xlabel('Pixel Values')
    plt.ylabel('Frequency')
    plt.show()

    threshold_value = 87
    thresholded_image = threshold_processing(gray_image, threshold_value)

    plt.imshow(thresholded_image, cmap='gray')
    plt.title(f'Thresholded image at threshold = {threshold_value}')
    plt.axis('off')
    plt.show()

    plt.imshow(gray_image, cmap='gray')
    plt.title(f'Thresholded image at threshold = {threshold_value}')
    plt.axis('off')
    plt.show()

    block_size = 16
    segments = segment_image(gray_image, block_size)

    max_histograms_per_figure = 30

    for batch_start in range(0, len(segments), max_histograms_per_figure):
        batch_end = min(batch_start + max_histograms_per_figure, len(segments))
        batch_segments = segments[batch_start:batch_end]

        num_segments = len(batch_segments)

        grid_size = int(np.ceil(np.sqrt(num_segments)))

        plt.figure(figsize=(12, 10))
        for i, segment in enumerate(batch_segments):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.hist(segment.ravel(), bins=256, range=(0, 256), color='black')
            plt.title(f'Segment {batch_start + i + 1}')
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.show()