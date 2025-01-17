# -- coding: utf-8 --
# @Time : 2024/4/27 14:11
# @Author : Stephanie
# @Email : sunc696@gmail.com
# @File : simulate.py
import numpy as np
import cv2
import os
import glob
from perlin_noise import PerlinNoise

# img_width = 800
# img_height = 600
base_image_path = 'D:/datasets/RICE_DATASET/RICE1/ground_truth/28.png'
cloud_mask_path = 'D:/datasets/RICE_DATASET/RICE1/mask/28.png'

def generate_perlin_noise(height, width, scale=10, octaves=6):
    noise = PerlinNoise(octaves=octaves, seed=1)
    pic = [[noise([i/scale, j/scale]) for j in range(width)] for i in range(height)]
    noise_array = np.array(pic)
    normalized_noise = np.interp(noise_array, (noise_array.min(), noise_array.max()), (0, 255))
    return normalized_noise.astype(np.uint8)

def apply_cloud_shadows(image, mask, noise, shadow_intensity=0.5):
    # Scale and threshold the Perlin noise to simulate cloud density
    shadow_strength = cv2.multiply(noise, mask)
    shadow_strength = cv2.normalize(shadow_strength, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Convert shadow strength to 3 channels
    shadow_strength = cv2.merge([shadow_strength] * 3)

    # Apply shadows to the image based on the Perlin noise and the cloud mask
    shadowed_image = cv2.subtract(image, shadow_strength.astype(np.uint8))
    shadowed_image = cv2.addWeighted(image, 1, shadowed_image, shadow_intensity, 0)
    return shadowed_image


# base_image = cv2.imread(base_image_path, cv2.IMREAD_COLOR)
#
# # Load the cloud mask image
# cloud_mask = cv2.imread(cloud_mask_path, cv2.IMREAD_GRAYSCALE)
#
# # Ensure the cloud mask is binary
# _, cloud_mask = cv2.threshold(cloud_mask, 127, 255, cv2.THRESH_BINARY)
#
# # Generate Perlin noise
# height, width = cloud_mask.shape
# perlin_noise = generate_perlin_noise(height, width, scale=50, octaves=6)
#
# # Apply the generated shadows to the image
# shadowed_image = apply_cloud_shadows(base_image, cloud_mask, perlin_noise, shadow_intensity=0.8)
#
# # Save the shadowed image
# shadowed_image_path = '1.jpg'
# cv2.imwrite(shadowed_image_path, shadowed_image)

# exit(0)


dataset_path = 'D:/datasets/RICE_DATASET/RICE2/ground_truth/'
output_path = 'D:/datasets/RICE_DATASET/RICE2/simulated-2/'
mask_path = 'D:/datasets/RICE_DATASET/RICE2/mask-1/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Get a list of all image files in the dataset directory
image_files = glob.glob(os.path.join(dataset_path, '30.png'))  # *.png

for image_file in image_files:
    # Read the base image and the mask
    base_image = cv2.imread(image_file, cv2.IMREAD_COLOR)

    # Construct the corresponding mask file path
    mask_file = os.path.join(mask_path, os.path.basename(image_file))
    cloud_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    # Check if the image and mask are loaded correctly
    if base_image is None or cloud_mask is None:
        print(f"Image or mask could not be loaded for {image_file}")
        continue

    # Ensure the cloud mask is binary
    _, cloud_mask = cv2.threshold(cloud_mask, 127, 255, cv2.THRESH_BINARY)

    # Generate Perlin noise
    height, width = cloud_mask.shape
    perlin_noise = generate_perlin_noise(height, width, scale=50, octaves=6)

    # Apply the generated shadows to the image
    shadowed_image = apply_cloud_shadows(base_image, cloud_mask, perlin_noise, shadow_intensity=0.4)
    # shadowed_image = apply_cloud_shadows(base_image, cloud_mask, perlin_noise, shadow_intensity=0.3)

    # Save the shadowed image
    output_file = os.path.join(output_path, os.path.basename(image_file))
    cv2.imwrite(output_file, shadowed_image)
    print(f"Processed image saved to {output_file}")

