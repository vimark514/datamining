import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(g_image, shannon_entropy_value, hartley_measure_value, markov_process_first_order_value,
                   channel):
    fig, (ax_hist, ax_image) = plt.subplots(1, 2, figsize=(12, 6))

    ax_hist.hist(g_image.ravel(), bins=256, range=(0, 256), density=True, color='gray')
    ax_hist.set_title(f'Histogram of Grayscale - Channel {channel}')
    ax_hist.set_xlabel('Pixel Intensity')
    ax_hist.set_ylabel('Frequency')

    fig.text(0.5, 0.15, f'Shannon Entropy: {shannon_entropy_value:.4f}',
             ha="center", fontsize=12)
    fig.text(0.5, 0.1, f'Hartley Measure: {hartley_measure_value:.4f}',
             ha="center", fontsize=12)
    fig.text(0.5, 0.05, f'Markov Process H: {markov_process_first_order_value:.4f}',
             ha="center", fontsize=12)

    ax_image.imshow(g_image, cmap='gray')
    ax_image.axis('off')
    ax_image.set_title(f'Original Image - Channel {channel}')

    plt.subplots_adjust(bottom=0.3)
    plt.show()


def plot_3d_bar_chart(shannon_values, hartley_values, markov_values):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    channels = ['R', 'G', 'B']
    x = np.arange(len(shannon_values))
    y_shannon = np.zeros_like(x)
    y_hartley = np.ones_like(x) * 1
    y_markov = np.ones_like(x) * 2
    z = np.zeros_like(x)
    width = 0.2
    depth = 0.2

    ax.bar3d(x - width, y_shannon, z, width, depth, shannon_values, color='r', label='Shannon Entropy')

    ax.bar3d(x, y_hartley, z, width, depth, hartley_values, color='g', label='Hartley Measure')

    ax.bar3d(x + width, y_markov, z, width, depth, markov_values, color='b', label='Markov Process')

    ax.set_xlabel('Channel')
    ax.set_ylabel('Metric')
    ax.set_zlabel('Value')
    ax.set_title('3D Bar Chart of Information Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Shannon', 'Hartley', 'Markov'])

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='r', label='Shannon Entropy'),
                       Patch(facecolor='g', label='Hartley Measure'),
                       Patch(facecolor='b', label='Markov Process')]
    ax.legend(handles=legend_elements)

    plt.show()
