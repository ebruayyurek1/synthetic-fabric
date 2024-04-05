import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.utils.img_utils import calculate_aspect_ratio_fit
from src.utils.io_utils import load_yaml


# import torch


class CanvasParameters:
    def __init__(self, config_path: str | Path):
        """
        Load a configuration file and create a simple object container

        :param config_path: str or Path to a configuration file (yaml)
        """
        # Load parameters
        parameters_dict: dict = load_yaml(config_path)

        # Resize logo
        self.logo_side = parameters_dict['logo_side_resize']

        # Specify grid dimensions
        self.rows: int = parameters_dict["rows"]
        self.cols: int = parameters_dict["cols"]

        # Specify distance between logos
        self.spacing: int = parameters_dict["spacing"]
        self.skew: float = parameters_dict["interval_skew_fraction"]

        # Specify transformation factors
        self.transform_prob: float = parameters_dict["transform_prob"]
        self.translation_factor: float = parameters_dict["translation_factor"]
        self.rotation_factor: float = parameters_dict["rotation_factor"]
        self.scale_factor: float = parameters_dict["scale_factor"]

        # Gaussian noise parameters (basically mean and std)
        self.min_noise: float = parameters_dict["min_noise"]
        self.max_noise: float = parameters_dict["max_noise"]

        # Background
        self.bg_color = parameters_dict["bg_color"]
        self.bg_texture = parameters_dict["bg_texture"]
        self.use_texture = parameters_dict["use_texture"]


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
    # Each dimension has spacing between each image (but not at the end) (+1 for image 0)
    canvas_width: int = c_params.cols * (cell_width + c_params.spacing) - c_params.spacing
    canvas_height: int = c_params.rows * (cell_height + c_params.spacing) - c_params.spacing
    # A blank canvas with either a parameter defined color background or repeated texture
    original_canvas: Image.Image = Image.new('RGBA', (canvas_width, canvas_height),
                                             tuple(c_params.bg_color))
    if c_params.use_texture:
        bg = Image.open(f"data/textures/{c_params.bg_texture}")
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
    sigma = sum(shape) / 10
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
    repeated_arr = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
    alpha_arr = np.ones((noise.shape[0], noise.shape[1], 1)) * 255
    noise = np.concatenate((repeated_arr, alpha_arr), axis=2)
    noisy_image_array = image_array + noise

    # Clip values within range
    noisy_image_array = np.clip(noisy_image_array, 0, 255)
    noisy_image = Image.fromarray(noisy_image_array.astype(np.uint8))
    return noisy_image


def run(main_folder: Path, input_img_name: str, c_params: CanvasParameters):
    # Note: all of these work on ints, not floating points.
    # Base path and image path
    base_output_path: Path = main_folder / 'output'
    image_path: Path = (main_folder / 'input' / input_img_name)
    # Load image
    logo: Image.Image = Image.open(main_folder / 'input' / input_img_name).convert('RGBA')
    # Create output folder
    logo_output_path: Path = base_output_path / image_path.stem
    logo_output_path.mkdir(parents=True, exist_ok=True)
    # Define cell size
    cell_height, cell_width, logo, original_canvas = make_background(c_params, logo)
    background = original_canvas.copy()
    # List to store center points before and after transformations
    center_points_before: list = []
    center_points_after: list = []

    # Paste the original logos onto the canvas with the original logos
    top_left_corners_before: list = generate_coordinates(original_canvas.width, original_canvas.height,
                                                         c_params.cols, c_params.rows, c_params.skew)

    for x_tl, y_tl in top_left_corners_before:
        # Put in the middle of spacing
        x_tl_w_space, y_tl_w_space = round(x_tl + c_params.spacing / 2), round(y_tl + c_params.spacing / 2)
        # box: 2-tuple giving the upper left corner
        original_canvas.paste(logo, (x_tl_w_space, y_tl_w_space), logo)
        # save (floating point) centers
        center_points_before.append((x_tl_w_space + logo.width / 2, y_tl_w_space + logo.height / 2))

    # Save the unaltered canvas
    unaltered_canvas_path: Path = logo_output_path / 'canvas_before.png'
    unaltered_canvas_path.parent.mkdir(exist_ok=True, parents=True)
    original_canvas.save(unaltered_canvas_path)
    # ------------------------------------------------------------------------
    # Generate random transformations
    # --- Translation ---
    t_f: float = c_params.translation_factor
    t_xs = np.random.uniform(-t_f * cell_width, t_f * cell_width, len(top_left_corners_before))
    t_ys = np.random.uniform(-t_f * cell_height, t_f * cell_height, len(top_left_corners_before))
    # --- Rotation ---
    r_f: float = c_params.rotation_factor
    angles: np.ndarray = np.random.uniform(-r_f * 360, r_f * 360, len(top_left_corners_before))
    # --- Scale ---
    s_f: float = c_params.scale_factor
    scale_factors: np.ndarray = np.random.uniform(1 - s_f, 1 + s_f, len(top_left_corners_before))
    # --- Probabilities ---
    p_t, p_r, p_s, p_n = np.random.uniform(0, 1, 4)
    # Generate random Gaussian noise amounts
    noise_amounts: np.ndarray[float] = np.random.uniform(c_params.min_noise, c_params.max_noise,
                                                  len(top_left_corners_before))
    for idx, (x_tl, y_tl) in enumerate(top_left_corners_before):
        # Random transformations
        # --- ROTATE ---
        if p_r >= 1 - c_params.transform_prob:
            # Reset probability to a random value
            p_r = np.random.uniform(0, 1)
            rotated_logo: Image.Image = logo.rotate(angles[idx], expand=True)
        else:
            # Additive probability to ensure it happens
            p_r += np.random.uniform(0, 1)
            rotated_logo: Image.Image = logo
        # ---  SCALE ---
        if p_s >= 1 - c_params.transform_prob:
            # Reset probability to a random value
            p_s = np.random.uniform(0, 1)
            scaled_logo: Image.Image = rotated_logo.resize(
                (round(rotated_logo.width * scale_factors[idx]), round(rotated_logo.height * scale_factors[idx])),
                resample=Image.Resampling.LANCZOS)
        else:
            # Additive probability to ensure it happens
            p_s += np.random.uniform(0, 1)
            scaled_logo: Image.Image = rotated_logo
        # --- TRANSLATION ---
        if p_t >= 1 - c_params.transform_prob:
            # Reset probability to a random value
            p_t = np.random.uniform(0, 1)
            x_tl_after: float = x_tl + t_xs[idx]
            y_tl_after: float = y_tl + t_ys[idx]
        else:
            # Additive probability to ensure it happens
            p_t += np.random.uniform(0, 1)
            x_tl_after: float = x_tl
            y_tl_after: float = y_tl
        # NOTE: Is there a way to do it float?
        # NOTE: Not really...
        # Putting logos onto the canvas with random transformations and random noise
        x = round(x_tl_after + c_params.spacing / 2)
        y = round(y_tl_after + c_params.spacing / 2)
        background.paste(scaled_logo, (x, y), scaled_logo)

        # Store the center points before and after translation
        center_points_after.append((x + scaled_logo.width / 2,
                                    y + scaled_logo.height / 2))

        # Gaussian noise to the logo
        # --- NOISE ---
        if p_n >= 1 - c_params.transform_prob:
            # Reset probability to a random value
            p_n = np.random.uniform(0, 1)
            background = add_centered_gaussian_noise(background, *center_points_after[-1], shape=scaled_logo.size,
                                                     noise_amount=noise_amounts[idx])
        else:
            # Additive probability to ensure it happens
            p_n += np.random.uniform(0, 1)
    # ------------------------------------------------------------------------
    # Save center coordinates
    centers_before = pd.DataFrame(center_points_before, columns=["x", "y"])
    centers_after = pd.DataFrame(center_points_after, columns=["x", "y"])
    centers_before.to_csv(logo_output_path / 'centers_before.csv', index=False)
    centers_after.to_csv(logo_output_path / 'centers_after.csv', index=False)

    background.save(logo_output_path / 'canvas_after.png')


def main():
    base_path: Path = Path('data')
    img_names: list[str] = ['instagram_logo.png', 'starbucks_logo.png', 'CocaCola_logo.png']
    for img_name in img_names:
        canvas_params = CanvasParameters(base_path / 'input' / 'parameters.yaml')
        times = []
        range_len = 1
        for i in range(range_len):
            start = time.perf_counter()
            run(main_folder=base_path, input_img_name=img_name, c_params=canvas_params)
            times.append(time.perf_counter() - start)
        print(f'Average time elapsed ({range_len} runs) for "{img_name}": {np.average(times):.2f}s')


if __name__ == "__main__":
    main()
