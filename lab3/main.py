import numpy as np
import cv2
import time

from imageEditor import *
from imageInfo import *
from outputBuilder import *

file_path = '../photos/I05.BMP'

channel_names = ['B', 'G', 'R']
image = cv2.imread(file_path)

if image is None:
    raise FileNotFoundError(f"Image not found: '{file_path}'")
else:
    noisy_image_gaussian = add_gaussian_noise_cv2(image)
    noisy_image_poisson = add_poisson_noise_cv2(image)
    noisy_image_speckle = add_speckle_noise_cv2(image)

    metrics = {}

    for noise_type, noisy_image in zip(['Gaussian', 'Poisson', 'Speckle'],
                                       [noisy_image_gaussian, noisy_image_poisson, noisy_image_speckle]):
        noisy_pixels_count = count_noisy_pixels(image, noisy_image)
        normalized_correlation = calculate_normalized_correlation(image, noisy_image)
        mse_value = mean_squared_error(image, noisy_image)
        psnr_value = calculate_psnr(image, noisy_image)
        max_error = calculate_max_error(image, noisy_image)

        metrics[noise_type] = {
            "Noisy Pixels Count": noisy_pixels_count,
            "Normalized Correlation": normalized_correlation.item(),
            "Mean Squared Error": mse_value.item(),
            "PSNR": psnr_value.item(),
            "Max Error": max_error.item()
        }

    for noise_type, result in metrics.items():
        print(f"{noise_type} Noise Metrics: {result}")

    show_two_images(image, noisy_image_gaussian, title1='Original Image', title2='Gaussian Noise')
    show_two_images(image, noisy_image_poisson, title1='Original Image', title2='Poisson Noise')
    show_two_images(image, noisy_image_speckle, title1='Original Image', title2='Speckle Noise')

    image_size_bytes = image.nbytes
    transmission_speeds = {
        "Fast Ethernet (100 Mbps)": 100.0,
        "Wi-Fi 802.11ac (1 Gbps)": 1000.0,
        "5G (10 Gbps)": 10000.0
    }
    for protocol, speed in transmission_speeds.items():
        transmission_time = calculate_transmission_time(image_size_bytes, speed)
        print(f"Protocol: {protocol}")
        print(f"Estimated transmission time: {transmission_time:.4f} seconds\n")

    snr_before_gaussian = calculate_snr(image.astype(np.float32), np.zeros_like(image).astype(np.float32))
    snr_after_gaussian = calculate_snr(image.astype(np.float32), (noisy_image_gaussian - image).astype(np.float32))

    snr_before_poisson = calculate_snr(image.astype(np.float32), np.zeros_like(image).astype(np.float32))
    snr_after_poisson = calculate_snr(image.astype(np.float32), (noisy_image_poisson - image).astype(np.float32))

    snr_before_speckle = calculate_snr(image.astype(np.float32), np.zeros_like(image).astype(np.float32))
    snr_after_speckle = calculate_snr(image.astype(np.float32), (noisy_image_speckle - image).astype(np.float32))

    print("SNR Results:")
    print(f"Gaussian Noise: SNR Before = {snr_before_gaussian:.2f} dB, SNR After = {snr_after_gaussian:.2f} dB")
    print(f"Poisson Noise: SNR Before = {snr_before_poisson:.2f} dB, SNR After = {snr_after_poisson:.2f} dB")
    print(f"Speckle Noise: SNR Before = {snr_before_speckle:.2f} dB, SNR After = {snr_after_speckle:.2f} dB")
