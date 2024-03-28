import random
import csv
from PIL import Image
import numpy as np

# Read the logo image
logo = Image.open('/Users/ebruayyurek/PycharmProjects/syntetic_fabric/data/output/instagram.png').convert('RGBA')

# Define grid dimensions
rows = 10
cols = 10

mean = 0
stddev = 25  # Adjust stddev as needed

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

# Create a blank canvas with a white background
original_canvas = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 255))

# Lists to store center points before and after transformations
center_points_before = []
center_points_after = []

# Paste the original logos onto the canvas with the original logos
for i in range(rows):
    for j in range(cols):
        x_center = j * (cell_width + spacing) + (cell_width + spacing) // 2
        y_center = i * (cell_height + spacing) + (cell_height + spacing) // 2
        original_canvas.paste(logo, (int(x_center - cell_width / 2),
                                     int(y_center - cell_height / 2)), logo)

# Save the canvas with original logos as the first image
original_canvas.save('original_canvas.png')

# ccreate a blank canvas with a white background
background = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 255))

# Putting logos onto the canvas with random transformations !!
for i in range(rows):
    for j in range(cols):
        # Calculate initial position for each logo (center of the grid)
        x_center_before = j * (cell_width + spacing) + (cell_width + spacing) // 2
        y_center_before = i * (cell_height + spacing) + (cell_height + spacing) // 2

        # Generate random transformations
        tx = random.uniform(-translation_percentage * cell_width, translation_percentage * cell_width)
        ty = random.uniform(-translation_percentage * cell_height, translation_percentage * cell_height)
        angle = random.uniform(-rotation_percentage * 360, rotation_percentage * 360)
        scale_factor = random.uniform(1 - scale_percentage, 1 + scale_percentage)

        # Apply transformations ....
        translated_logo = logo.transform(logo.size, Image.AFFINE, (1, 0, tx, 0, 1, ty))
        rotated_and_scaled_logo = translated_logo.rotate(angle, expand=True).resize(
            (int(logo.width * scale_factor), int(logo.height * scale_factor)))

        # Calculate final position of the center points after transformations here just in translationn
        x_center_after = x_center_before + tx
        y_center_after = y_center_before + ty

        # Place the transformed logo onto the canvas
        background.paste(rotated_and_scaled_logo, (int(x_center_after - rotated_and_scaled_logo.width / 2),
                                                   int(y_center_after - rotated_and_scaled_logo.height / 2)),
                         rotated_and_scaled_logo)


        # sstore the center points before and after translation
        center_points_before.append((x_center_before, y_center_before))
        center_points_after.append((x_center_after, y_center_after))

"""
# apply Gaussian noise to the final image ?
img_array = np.array(background)
noise = np.random.normal(mean, stddev, img_array.shape[:-1])
noisy_background = Image.fromarray(np.uint8(np.clip(img_array + noise[:, :, np.newaxis], 0, 255)))

"""

# Save center points before and after transformations to CSV file
with open('center_points_before_after.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['X Before', 'Y Before', 'X After', 'Y After'])
    for i in range(len(center_points_before)):
        writer.writerow([center_points_before[i][0], center_points_before[i][1],
                         center_points_after[i][0], center_points_after[i][1]])

# Save the final image with transformed logos
#noisy_background.save('/Users/ebruayyurek/Documents/synthetic_image/data/output/final_image_trial.png')
background.save('/Users/ebruayyurek/Documents/synthetic_image/data/output/final_image_trial.png')




