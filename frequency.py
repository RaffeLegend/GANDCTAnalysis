import os
import numpy as np
from PIL import Image

from src.image_np import dct2, load_image, normalize, scale_image, fft2d
from src.math import log_scale, welford
from scipy import fftpack

import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm

def transform(image):
    # image = _dct2_wrap(image)
    image = fft2d(image)
    return image

def calculate_absolute_value(images):
    first = images[0]
    current_max = np.absolute(first)
    progress_bar = tqdm(total=len(images), desc="absolute")
    for data in images:
        progress_bar.update(1)
        max_values = np.absolute(data)
        mask = current_max > max_values
        current_max *= mask
        current_max += max_values * ~mask
    progress_bar.close()
    return current_max

def scale_by_absolute(image, current_max):
    return image / current_max

def calculate_mean_std(images):
    mean, var = welford(images)
    std = np.sqrt(var)
    return mean, std

def load_images_from_folder(image_folder, max_images=200):
    images = []
    count = 0
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            if max_images is not None and count >= max_images:
                break
            image_path = os.path.join(image_folder, filename)
            image = load_image(image_path, grayscale=True)
            images.append(np.array(image))
            count += 1
    return images

def calculate_average_frequency(images, absolute_value, mean, std):
    frequencies = []
    count = 0
    progress_bar = tqdm(total=len(images), desc="calculating")
    for image_array in images:
        progress_bar.update(1)
        image_array = transform(image_array)
        image_array = scale_by_absolute(image_array, absolute_value)
        image_array = normalize(image_array, mean, std)

        resized_magnitude_spectrum = cv2.resize(image_array, (512, 512))
        
        if frequency_sum is None:
            frequency_sum = np.zeros_like(resized_magnitude_spectrum)
        
        frequency_sum += resized_magnitude_spectrum
        count += 1
    
    progress_bar.close()
    average_frequency = frequency_sum / count
    return average_frequency

def save_average_frequency_image(average_frequency, output_path):
    
    # Normalize the average frequency to 0-255
    # normalized_frequency = np.uint8(255 * (average_frequency - np.min(average_frequency)) / (np.max(average_frequency) - np.min(average_frequency)))
    normalized_frequency = np.uint8(255 * average_frequency)
    # Convert to color image
    color_average_frequency = cv2.applyColorMap(normalized_frequency, cv2.COLORMAP_JET)
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(color_average_frequency, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    color_average_frequency = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Save the result image
    output_name = 'average_frequency.png'
    output_file_path = os.path.join(output_path, output_name)
    cv2.imwrite(output_file_path, color_average_frequency)
    print(f"Average frequency image saved at {output_file_path}")


if __name__ == "__main__":
    image_folder = '/path/to/your/image/folder'
    output_folder = './'
    images = load_images_from_folder(image_folder)
    absolute_value = calculate_absolute_value(images)
    mean, std = calculate_mean_std(images)
    average_frequency = calculate_average_frequency(images, absolute_value, mean, std)
    save_average_frequency_image(average_frequency, output_folder)