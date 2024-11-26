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


def dct_2d(image):
    h, w = image.shape
    dct_matrix = np.zeros((h, w))
    for u in range(h):
        for v in range(w):
            sum_val = 0
            for x in range(h):
                for y in range(w):
                    sum_val += image[x, y] * np.cos(((2 * x + 1) * u * np.pi) / (2 * h)) * \
                               np.cos(((2 * y + 1) * v * np.pi) / (2 * w))
            c_u = np.sqrt(1 / h) if u == 0 else np.sqrt(2 / h)
            c_v = np.sqrt(1 / w) if v == 0 else np.sqrt(2 / w)
            dct_matrix[u, v] = c_u * c_v * sum_val
    return dct_matrix


def idct_2d(dct_matrix):
    h, w = dct_matrix.shape
    image_reconstructed = np.zeros((h, w))
    for x in range(h):
        for y in range(w):
            sum_val = 0
            for u in range(h):
                for v in range(w):
                    c_u = np.sqrt(1 / h) if u == 0 else np.sqrt(2 / h)
                    c_v = np.sqrt(1 / w) if v == 0 else np.sqrt(2 / w)
                    sum_val += c_u * c_v * dct_matrix[u, v] * np.cos(((2 * x + 1) * u * np.pi) / (2 * h)) * \
                               np.cos(((2 * y + 1) * v * np.pi) / (2 * w))
            image_reconstructed[x, y] = sum_val
    return image_reconstructed


def reconstruct_segment_dct(segment):
    dct_coefficients = dct_2d(segment)
    dct_coefficients_rounded = np.round(dct_coefficients)
    reconstructed_segment = idct_2d(dct_coefficients_rounded)
    mse = np.mean((segment - reconstructed_segment) ** 2)

    dct_coeff_magnitude = np.mean(np.abs(dct_coefficients))

    return reconstructed_segment, mse, dct_coeff_magnitude


def reconstruct_combined_image(img, segments, positions, segment_size=8):
    reconstructed_img = np.zeros_like(img)
    for category in segments.keys():
        for segment, (i, j) in zip(segments[category], positions[category]):
            reconstructed_segment, _, _ = reconstruct_segment_dct(segment)
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


def display_segment_reconstruction(segment):
    reconstructed_segment, mse, dct_coeff_magnitude = reconstruct_segment_dct(segment)

    mse_rounded = round(mse, 4)
    dct_coeff_magnitude_rounded = int(dct_coeff_magnitude)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(segment, cmap='gray')
    axes[0].set_title("Original Segment")
    axes[0].axis("off")

    axes[1].imshow(reconstructed_segment, cmap='gray')
    axes[1].set_title(f"Reconstructed Segment, MSE={mse_rounded}, DCT Magnitude={dct_coeff_magnitude_rounded}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    return reconstructed_segment


def display_image(image, title="Image"):
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis("off")
    plt.show()


def create_dct_heatmap(image, segment_size=8):
    h, w = image.shape
    heatmap = np.zeros((h, w))

    for i in range(0, h - segment_size + 1, segment_size):
        for j in range(0, w - segment_size + 1, segment_size):
            segment = image[i:i + segment_size, j:j + segment_size]
            dct_coefficients = dct_2d(segment)
            dct_magnitude = np.mean(np.abs(dct_coefficients))

            heatmap[i:i + segment_size, j:j + segment_size] = dct_magnitude

    return heatmap


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

        axes[i].plot(reconstructed_values, label='Reconstructed', color='red', linestyle='-', marker='o')

        axes[i].set_title(f'{axis.capitalize()} {i}')
        axes[i].legend()

    for j in range(n_plots, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def display_heatmap(heatmap):
    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title("DCT Coefficient Heatmap (Grayscale)")
    plt.axis("off")
    plt.show()


def calculate_average_dct_magnitude_for_categories(segments, positions):
    avg_dct_magnitudes = {"low": [], "middle": [], "high": []}

    for category in segments.keys():
        total_magnitude = 0
        count = 0
        for segment, (i, j) in zip(segments[category], positions[category]):
            _, _, dct_coeff_magnitude = reconstruct_segment_dct(segment)
            total_magnitude += dct_coeff_magnitude
            count += 1
        avg_magnitude = total_magnitude / count if count > 0 else 0
        avg_dct_magnitudes[category] = avg_magnitude

    return avg_dct_magnitudes


def calculate_mse_for_segments(segments, positions):
    mse_values = {"low": [], "middle": [], "high": []}

    for category in segments.keys():
        for segment, (i, j) in zip(segments[category], positions[category]):
            _, mse, _ = reconstruct_segment_dct(segment)
            mse_values[category].append(mse)

    return mse_values


def plot_mse_vs_category(mse_values):
    categories = ['low', 'middle', 'high']
    avg_mse = [np.mean(mse_values[category]) for category in categories]

    plt.figure(figsize=(8, 6))
    plt.bar(categories, avg_mse, color=['blue', 'orange', 'green'])
    plt.xlabel('Category (Entropy)')
    plt.ylabel('Average MSE')
    plt.title('Average MSE vs Entropy Category')
    plt.show()


if __name__ == "__main__":
    image = get_image_from(IMAGE_PATH)
    image = get_grayscale_image_of(image)
    display_image(image, title="Grayscale Image")

    segments, positions = classify_segments(image, segment_size=SEGMENT_SIZE)

    mse_values = calculate_mse_for_segments(segments, positions)

    plot_mse_vs_category(mse_values)

# if __name__ == "__main__":
#     image = get_image_from(IMAGE_PATH)
#     image = get_grayscale_image_of(image)
#     display_image(image, title="Grayscale Image")
#
#     segments, positions = classify_segments(image, segment_size=SEGMENT_SIZE)
#
#     # Visualize segments based on entropy categories
#     visualize_segments(image, segments, positions, segment_size=SEGMENT_SIZE)
#
#     # Create and display DCT heatmap
#     dct_heatmap = create_dct_heatmap(image, segment_size=SEGMENT_SIZE)
#     display_heatmap(dct_heatmap)
#
#     avg_dct_magnitudes = calculate_average_dct_magnitude_for_categories(segments, positions)
#     print("Average DCT Magnitude for each category:")
#     print(f"Low Entropy: {avg_dct_magnitudes['low']}")
#     print(f"Middle Entropy: {avg_dct_magnitudes['middle']}")
#     print(f"High Entropy: {avg_dct_magnitudes['high']}")
#
#     plot_reconstruction(original_segment=segments["low"][0],
#                         reconstructed_segment=display_segment_reconstruction(segments["low"][0]),
#                         axis='row')
#
#     plot_reconstruction(original_segment=segments["middle"][0],
#                         reconstructed_segment=display_segment_reconstruction(segments["middle"][0]),
#                         axis='row')
#
#     plot_reconstruction(original_segment=segments["high"][0],
#                         reconstructed_segment=display_segment_reconstruction(segments["high"][0]),
#                         axis='row')
#
#     reconstructed_image = reconstruct_combined_image(image, segments, positions, segment_size=SEGMENT_SIZE)
#     display_image(reconstructed_image, title="Reconstructed Image")
