import cv2
import numpy as np
from PIL import Image

from src.data_generation.core.CanvasParameters import CanvasParameters
from src.utils.img_utils import calculate_aspect_ratio_fit


# import torch


def generate_coordinates(width: int, height: int,
                         width_intervals: int, height_intervals: int,
                         interval_skew_fraction: float) -> list[[int, int]]:
    # Calculate interval sizes
    x_interval = width / width_intervals
    y_interval = height / height_intervals

    coordinates = []
    for j in range(height_intervals):
        skew = 0 if j % 2 == 0 else interval_skew_fraction * x_interval
        # - j % 2 removes one on the skewed row
        for i in range(width_intervals - (j % 2 if interval_skew_fraction > 0.1 else 0)):
            coordinates.append((int(i * x_interval + skew), int(j * y_interval)))

    return coordinates


def make_background(c_params, logo):
    if c_params.logo_side != -1:
        cell_width, cell_height = calculate_aspect_ratio_fit(logo.width, logo.height, c_params.logo_side)
        logo: Image.Image = logo.resize((cell_width, cell_height))
    else:
        cell_width: int = logo.width
        cell_height: int = logo.height
    # Dimension of entire canvas
    # Each dimension has spacing between each image
    canvas_width: int = round((c_params.cols ) * (cell_width + c_params.spacing))
    canvas_height: int = round((c_params.rows) * (cell_height + c_params.spacing))
    # A blank canvas with either a parameter defined color background or repeated texture
    original_canvas: Image.Image = Image.new('RGB', (canvas_width, canvas_height),
                                             tuple(c_params.bg_color))
    # Paste texture if present
    if c_params.use_texture:
        bg = Image.open(c_params.bg_texture)
        bg_w, bg_h = bg.size
        # Iterate through a grid, to place the background tile
        for i in range(0, canvas_width, bg_w):
            for j in range(0, canvas_height, bg_h):
                # paste the image at location i, j:
                original_canvas.paste(bg, (i, j))
    return cell_height, cell_width, logo, original_canvas


def add_centered_gaussian_noise(image, center_x, center_y, shape: tuple[int, int],
                                noise_amount: float = 100.0):
    # TODO: arbitrary
    sigma = sum(shape) / 5
    logo_width, logo_height = shape
    tx = np.random.uniform(- logo_width / 3, logo_width / 3)
    ty = np.random.uniform(- logo_height / 3, logo_height / 3)
    image_array = np.array(image)
    # --- Calculate distances ---
    x_indices, y_indices = np.meshgrid(np.arange(image.height), np.arange(image.width), indexing='ij')
    y_coord, x_coord = center_x + tx, center_y + ty
    squared_dist = (x_indices - x_coord) ** 2 + (y_indices - y_coord) ** 2
    gaussian_shape = np.exp(-squared_dist / (2 * sigma ** 2))
    # --- Create noise ---
    noise = np.random.normal(scale=noise_amount, size=image.size[::-1]) * gaussian_shape
    noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
    # alpha_arr = np.ones((noise.shape[0], noise.shape[1], 1)) * 255
    # noise = np.concatenate((repeated_arr, alpha_arr), axis=2)
    noisy_image_array = image_array + noise

    # Clip values within range
    noisy_image_array = np.clip(noisy_image_array, 0, 255)
    noisy_image = Image.fromarray(noisy_image_array.astype(np.uint8))
    return noisy_image


def add_centered_gaussian_blur(image, center_x, center_y, shape: tuple[int, int],
                               blur_kernel: [int, int] = (5, 5)):
    logo_width, logo_height = shape
    # Random shift in center
    tx = np.random.uniform(- logo_width / 3, logo_width / 3)
    ty = np.random.uniform(- logo_height / 3, logo_height / 3)
    blurred_img = cv2.GaussianBlur(np.array(image), blur_kernel, 0)
    # --- Calculate distances ---

    # --- Create blur ---
    w, h = image.width, image.height
    Y, X = np.ogrid[:h, :w]
    y_coord, x_coord = center_x + tx, center_y + ty
    dist_from_center = np.sqrt((X - x_coord) ** 2 + (Y - y_coord) ** 2)

    # Start the gradient at the specified "radius"
    # TODO: arbitrary
    small_radius = min(logo_width, logo_height) / 2
    mask = np.clip((dist_from_center - small_radius) / small_radius, 0, 1)

    # Repeat mask across channels
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Blend the original and blurred image using the mask
    blended_img = mask * image + (1 - mask) * blurred_img
    # Clip values within range
    blended_img_array = np.clip(blended_img, 0, 255)
    blended_img = Image.fromarray(blended_img_array.astype(np.uint8))
    return blended_img


def soft_blur_img(img: np.ndarray, radius: int, scale: int = 1, **blur_params):
    w, h, _ = img.shape
    rescaled_radius = int(radius * scale)
    square_center_coordinates = ((w // 2) * scale, (h // 2) * scale)
    blurred_img = cv2.GaussianBlur(img, **blur_params)

    # Create a radial gradient mask
    Y, X = np.ogrid[:h, :w]  # Grid of Y, X indices
    # Matrix of distances to center
    dist_from_center = np.sqrt((X - square_center_coordinates[0]) ** 2 + (Y - square_center_coordinates[1]) ** 2)

    # Start the gradient at the specified "radius"
    mask = np.clip((dist_from_center - rescaled_radius) / rescaled_radius, 0, 1)

    # Repeat mask across channels
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Blend the original and blurred image using the mask
    blended_img = (1.0 - mask) * img + mask * blurred_img
    return blended_img.astype(np.uint8)


def change_contrast_in_bands(image, c_params: CanvasParameters):
    width, height = image.size
    band_height = int(height * c_params.band_percentage)
    for y in range(0, height, band_height):
        contrast_factor = np.random.uniform(c_params.contrast_min, c_params.contrast_max)
        box = (0, y, width, min(y + band_height, height))
        region = image.crop(box)
        region = region.point(lambda p: p * contrast_factor)
        image.paste(region, box)
    return image
