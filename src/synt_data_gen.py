import random
import csv
from PIL import Image
import numpy as np

#  logo image
logo = Image.open('/Users/ebruayyurek/PycharmProjects/syntetic_fabric/data/input/instagram_logo.png').convert('RGBA')

# grid dimensions
rows = 10
cols = 10

# Cell size
cell_width = logo.width
cell_height = logo.height
spacing = 50  # distance between logos

# Canvas size
canvas_width = cols * (cell_width + spacing) - spacing
canvas_height = rows * (cell_height + spacing) - spacing

# Transformation percentages
translation_percentage = 0.1  # 10% translation
rotation_percentage = 0.03  # 3% rotation
scale_percentage = 0.01  # 10% scaling

# gaussian noise parameters(basically mean and std)
min_noise = 0
max_noise = 0.5

#a blank canvas with a white background
original_canvas = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 255))

#  list store center points before and after transformations
center_points_before = []
center_points_after = []

# Paste the original logos onto the canvas with the original logos
for i in range(rows):
    for j in range(cols):
        x_center = j * (cell_width + spacing) + (cell_width + spacing) // 2
        y_center = i * (cell_height + spacing) + (cell_height + spacing) // 2
        original_canvas.paste(logo, (int(x_center - cell_width / 2),
                                     int(y_center - cell_height / 2)), logo)

# saving canvas with original logos as the first image
original_canvas.save('original_canvas.png')

# creati a blank canvas with a white background
background = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 255))

# Putting logos onto the canvas with random transformations and random noise
for i in range(rows):
    for j in range(cols):
        # Calculate initial position for each logo (center of the grid)
        x_center_before = j * (cell_width + spacing) + (cell_width + spacing) // 2
        y_center_before = i * (cell_height + spacing) + (cell_height + spacing) // 2

        # Generate random Gaussian noise amount that I parametrised before
        noise_amount = random.uniform(min_noise, max_noise)

        #  Gaussian noise to the logo
        noisy_logo = Image.new('RGBA', logo.size)
        pixels = np.array(logo)
        noise = np.random.normal(0, noise_amount, pixels.shape).astype(np.uint8)
        noisy_pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8) # ensures that the pixel values stay within the valid range of 0 to 255 (inclusive), preventing overflow or underflow. Any resulting values below 0 are set to 0, and any values above 255 are set to 255.
        noisy_logo = Image.fromarray(noisy_pixels, 'RGBA')

        # random transformations
        tx = random.uniform(-translation_percentage * cell_width, translation_percentage * cell_width)
        ty = random.uniform(-translation_percentage * cell_height, translation_percentage * cell_height)
        angle = random.uniform(-rotation_percentage * 360, rotation_percentage * 360)
        scale_factor = random.uniform(1 - scale_percentage, 1 + scale_percentage)

        # Apply transformations ....
        translated_logo = noisy_logo.transform(noisy_logo.size, Image.AFFINE, (1, 0, tx, 0, 1, ty))
        rotated_and_scaled_logo = translated_logo.rotate(angle, expand=True).resize(
            (int(noisy_logo.width * scale_factor), int(noisy_logo.height * scale_factor)))

        # final position of the center points after transformations
        x_center_after = x_center_before + tx
        y_center_after = y_center_before + ty

        # place the transformed logo onto the canvas
        background.paste(rotated_and_scaled_logo, (int(x_center_after - rotated_and_scaled_logo.width / 2),
                                                   int(y_center_after - rotated_and_scaled_logo.height / 2)),
                         rotated_and_scaled_logo)

        # Store the center points before and after translation
        center_points_before.append((x_center_before, y_center_before))
        center_points_after.append((x_center_after, y_center_after))
"""
# Save center points before and after transformations to CSV file
with open('center_points_before_after.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['X Before', 'Y Before', 'X After', 'Y After'])
    for i in range(len(center_points_before)):
        writer.writerow([center_points_before[i][0], center_points_before[i][1],
                         center_points_after[i][0], center_points_after[i][1]])

"""
# save the final image with transformed logos
background.save('/Users/ebruayyurek/PycharmProjects/syntetic_fabric/data/output/final_image_withnoise.png')











