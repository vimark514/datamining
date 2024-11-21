import numpy as np
import matplotlib.pyplot as plt
from lab5.main import get_image_from
from lab6.main import get_grayscale_image_of

SEGMENT_SIZE = 8
IMAGE_PATH = '../photos/I05.BMP'


def calculate_shannon_entropy(hist, base=2):
    hist = hist[hist > 0]
    probabilities = hist / np.sum(hist)
    entropy = -np.sum(probabilities * np.log(probabilities) / np.log(base))
    return entropy


def classify_segments(img, segment_size=8):
    segments = {"low": [], "middle": [], "high": []}
    positions = {"low": [], "middle": [], "high": []}
    entropies = []
    h, w = img.shape

    for i in range(0, h - segment_size + 1, segment_size):
        for j in range(0, w - segment_size + 1, segment_size):
            segment = img[i:i + segment_size, j:j + segment_size]
            flat_segment = segment.flatten()
            hist, _ = np.histogram(flat_segment, bins=256, range=(0, 255), density=True)
            seg_entropy = calculate_shannon_entropy(hist, base=2)
            entropies.append((seg_entropy, segment, (i, j)))

    entropies.sort(key=lambda x: x[0])
    low_thresh = entropies[int(len(entropies) * 0.33)][0]
    high_thresh = entropies[int(len(entropies) * 0.66)][0]

    for seg_entropy, segment, (i, j) in entropies:
        if seg_entropy <= low_thresh:
            segments["low"].append(segment)
            positions["low"].append((i, j))
        elif seg_entropy <= high_thresh:
            segments["middle"].append(segment)
            positions["middle"].append((i, j))
        else:
            segments["high"].append(segment)
            positions["high"].append((i, j))

    return segments, positions


def linear_regression_with_mse(x, y):
    a, b = np.polyfit(x, y, deg=1)
    y_pred = a * x + b
    mse = np.mean((y - y_pred) ** 2)
    return y_pred, mse


def reconstruct_segment_with_mse(segment, by_row=True):
    reconstructed_segment = np.zeros_like(segment)
    mse_total = 0
    n = segment.shape[0] if by_row else segment.shape[1]

    for i in range(n):
        x = np.arange(segment.shape[1] if by_row else segment.shape[0])
        y = segment[i] if by_row else segment[:, i]
        y_pred, mse = linear_regression_with_mse(x, y)
        if by_row:
            reconstructed_segment[i] = y_pred
        else:
            reconstructed_segment[:, i] = y_pred
        mse_total += mse

    avg_mse = mse_total / n
    return reconstructed_segment, avg_mse


def reconstruct_combined_image(img, segments, positions, segment_size=8, by_row=True):
    reconstructed_img = np.zeros_like(img)
    for category in segments.keys():
        for segment, (i, j) in zip(segments[category], positions[category]):
            reconstructed_segment, _ = reconstruct_segment_with_mse(segment, by_row)
            reconstructed_img[i:i + segment_size, j:j + segment_size] = reconstructed_segment

    return reconstructed_img


def visualize_segments(image, segments, positions, segment_size=8):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    entropy_categories = ["low", "middle", "high"]
    titles = ["Low Entropy", "Middle Entropy", "High Entropy"]

    for ax, category, title in zip(axes, entropy_categories, titles):
        visualization = np.zeros_like(image)
        for segment, (i, j) in zip(segments[category], positions[category]):
            visualization[i:i + segment_size, j:j + segment_size] = segment
        ax.imshow(visualization, cmap='gray')
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def display_segment_reconstruction(segment, by_row=True):
    reconstructed_segment, mse = reconstruct_segment_with_mse(segment, by_row)
    mse_rounded = round(mse, 4)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(segment, cmap='gray')
    axes[0].set_title("Original Segment")
    axes[0].axis("off")

    axes[1].imshow(reconstructed_segment, cmap='gray')
    axes[1].set_title(f"Reconstructed Segment, MSE={mse_rounded}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_reconstruction(original_segment, reconstructed_segment, axis='row'):
    if axis not in ['row', 'column']:
        raise ValueError("Axis must be either 'row' or 'column'")

    n_plots = original_segment.shape[0] if axis == 'row' else original_segment.shape[1]
    n_rows, n_cols = 2, 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(n_plots):
        original_values = original_segment[i, :] if axis == 'row' else original_segment[:, i]
        reconstructed_values = reconstructed_segment[i, :] if axis == 'row' else reconstructed_segment[:, i]

        axes[i].plot(original_values, label='Original', color='blue')
        axes[i].plot([0, len(reconstructed_values) - 1], [reconstructed_values[0], reconstructed_values[-1]],
                     label='Reconstructed', color='red', linestyle='-', marker='o')
        axes[i].set_title(f'{axis.capitalize()} {i}')
        axes[i].legend()

    for j in range(n_plots, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def display_image(image, title="Image"):
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    image = get_image_from(IMAGE_PATH)
    display_image(image, title="Image")
    image = get_grayscale_image_of(image)
    display_image(image, title="Grayscale image")
    segments, positions = classify_segments(image, segment_size=SEGMENT_SIZE)

    visualize_segments(image, segments, positions, segment_size=SEGMENT_SIZE)

    reconstructed_image_col = reconstruct_combined_image(image, segments, positions, segment_size=SEGMENT_SIZE,
                                                         by_row=False)
    display_image(reconstructed_image_col, title="Reconstructed Image (Column)")

    reconstructed_image_row = reconstruct_combined_image(image, segments, positions, segment_size=SEGMENT_SIZE,
                                                         by_row=True)
    display_image(reconstructed_image_row, title="Reconstructed Image (Row)")

    display_segment_reconstruction(segments["low"][0])
    display_segment_reconstruction(segments["middle"][0])
    display_segment_reconstruction(segments["high"][0])

    plot_reconstruction(segments["low"][0], reconstruct_segment_with_mse(segments["low"][0], by_row=False)[0],
                        axis='column')
    plot_reconstruction(segments["middle"][10], reconstruct_segment_with_mse(segments["middle"][10], by_row=False)[0],
                        axis='column')
    plot_reconstruction(segments["high"][20], reconstruct_segment_with_mse(segments["high"][20], by_row=False)[0],
                        axis='column')
