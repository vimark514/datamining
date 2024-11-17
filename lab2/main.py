import cv2

from imageEditor import *
from entropy import *
from hystogramBuilder import *

file_path = '../photos/I05.BMP'

channel_names = ['B', 'G', 'R']
image = cv2.imread(file_path)

if image is None:
    raise FileNotFoundError(f"Image not found: '{file_path}'")
else:
    def process_image_channels(image):
        channels = cv2.split(image)

        results = {}
        for i, channel in enumerate(channel_names):
            g_image = channels[i]
            shannon_value = shannon_entropy(g_image)
            hartley_value = hartley_measure(g_image)
            markov_value = markov_process_first_order(g_image)

            results[channel] = {
                'Shannon': shannon_value,
                'Hartley': hartley_value,
                'Markov': markov_value
            }

            plot_histogram(g_image, shannon_value, hartley_value, markov_value, channel)

        return results


    results = process_image_channels(image)

    shannon_values = [metrics['Shannon'] for metrics in results.values()]
    hartley_values = [metrics['Hartley'] for metrics in results.values()]
    markov_values = [metrics['Markov'] for metrics in results.values()]

    plot_3d_bar_chart(shannon_values, hartley_values, markov_values)

    segment_size = 256
    segments = segment_rgb_image(image, segment_size)

    metrics_by_channel = {channel: {'Shannon': [], 'Hartley': [], 'Markov': []} for channel in channel_names}

    for segment in segments:
        metrics = process_image_channels(segment)  # Get results for the current segment
        for channel in channel_names:
            metrics_by_channel[channel]['Shannon'].append(metrics[channel]['Shannon'])
            metrics_by_channel[channel]['Hartley'].append(metrics[channel]['Hartley'])
            metrics_by_channel[channel]['Markov'].append(metrics[channel]['Markov'])

        segment_shannon = [metrics[channel]['Shannon'] for channel in channel_names]
        segment_hartley = [metrics[channel]['Hartley'] for channel in channel_names]
        segment_markov = [metrics[channel]['Markov'] for channel in channel_names]

        plot_3d_bar_chart(segment_shannon, segment_hartley, segment_markov)

    for channel in channel_names:
        print(np.mean(metrics_by_channel[channel]['Shannon']))
        print(np.mean(metrics_by_channel[channel]['Hartley']))
        print(np.mean(metrics_by_channel[channel]['Markov']))

    shannon_means = [np.mean(metrics_by_channel[channel]['Shannon']) for channel in channel_names]
    hartley_means = [np.mean(metrics_by_channel[channel]['Hartley']) for channel in channel_names]
    markov_means = [np.mean(metrics_by_channel[channel]['Markov']) for channel in channel_names]

    plot_3d_bar_chart(shannon_means, hartley_means, markov_means)
