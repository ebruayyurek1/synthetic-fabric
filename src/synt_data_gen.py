import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from PIL.Image import Image as PILImage

from src.utils.img_utils import calculate_aspect_ratio_fit
from src.utils.io_utils import load_yaml


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

        # Specify transformation factors
        self.translation_factor: float = parameters_dict["translation_factor"]
        self.rotation_factor: float = parameters_dict["rotation_factor"]
        self.scale_factor: float = parameters_dict["scale_factor"]

        # Gaussian noise parameters (basically mean and std)
        self.min_noise: float = parameters_dict["min_noise"]
        self.max_noise: float = parameters_dict["max_noise"]

        # Background
        self.bg_color = parameters_dict["bg_color"]


def generate_coordinates(width, height, width_intervals, height_intervals):
    # Calculate interval sizes
    x_interval = width / width_intervals
    y_interval = height / height_intervals

    # Generate x-coordinates
    x_coords = [int(i * x_interval) for i in range(width_intervals)]

    # Generate y-coordinates
    y_coords = [int(i * y_interval) for i in range(height_intervals)]

    # Combine x and y coordinates
    coordinates = [(x, y) for x in x_coords for y in y_coords]

    return coordinates


def run(main_folder: Path, input_img_name: str, c_params: CanvasParameters):
    base_output_path: Path = main_folder / 'output'
    # Load the logo image
    # TODO: pass an image directly ? Re-think
    logo: PILImage = Image.open(main_folder / 'input' / input_img_name).convert('RGBA')

    # Define cell size
    if c_params.logo_side != -1:
        cell_width, cell_height = calculate_aspect_ratio_fit(logo.width, logo.height, c_params.logo_side)
        logo: PILImage = logo.resize((cell_width, cell_height))
    else:
        cell_width: int = logo.width
        cell_height: int = logo.height
    # Dimension of entire canvas
    # Each dimension has spacing between each image (but not at the end) (+1 for image 0)
    canvas_width: int = c_params.cols * (cell_width + c_params.spacing) - c_params.spacing
    canvas_height: int = c_params.rows * (cell_height + c_params.spacing) - c_params.spacing

    # A blank canvas with a white background
    original_canvas: PILImage = Image.new('RGBA', (canvas_width, canvas_height),
                                          tuple(c_params.bg_color))

    # List to store center points before and after transformations
    center_points_before: list = []
    center_points_after: list = []

    # Paste the original logos onto the canvas with the original logos
    # TODO: shift based on angle
    top_left_corners_before: list = generate_coordinates(canvas_width, canvas_height, c_params.rows, c_params.cols)

    for x_tl, y_tl in top_left_corners_before:
        # box: 2-tuple giving the upper left corner
        original_canvas.paste(logo, (x_tl + c_params.spacing // 2, y_tl + c_params.spacing // 2), logo)
        center_points_before.append((x_tl + logo.width // 2, y_tl + logo.height // 2))

    # Save the unaltered canvas
    unaltered_canvas_path: Path = base_output_path / 'canvas_before.png'
    unaltered_canvas_path.parent.mkdir(exist_ok=True, parents=True)
    original_canvas.save(unaltered_canvas_path)

    # A second blank canvas with a white background
    background: PILImage = Image.new('RGBA', (canvas_width, canvas_height), tuple(c_params.bg_color))
    # ------------------------------------------------------------------------
    # Putting logos onto the canvas with random transformations and random noise
    for x_tl, y_tl in top_left_corners_before:
        # Generate random Gaussian noise amount that I parametrised before
        noise_amount: float = random.uniform(c_params.min_noise, c_params.max_noise)

        # Gaussian noise to the logo
        pixels: np.ndarray = np.array(logo)
        noise: np.ndarray = np.random.normal(0, noise_amount, pixels.shape).astype(np.uint8)
        # Ensures that the pixel values stay within the valid range (inclusive).
        noisy_pixels: np.ndarray = np.clip(pixels + noise, 0, 255).astype(np.uint8)
        noisy_logo: PILImage = Image.fromarray(noisy_pixels, 'RGBA')

        # Random transformations
        # TODO: check magnitudes and what random uniform does
        # TODO: Maybe can simplify with one draw of multiple values
        t_f: float = c_params.translation_factor
        tx: float = random.uniform(-t_f * cell_width, t_f * cell_width)
        ty: float = random.uniform(-t_f * cell_height, t_f * cell_height)

        r_f: float = c_params.rotation_factor
        angle: float = random.uniform(-r_f * 360, r_f * 360)

        s_f: float = c_params.scale_factor
        scale_factor: float = random.uniform(1 - s_f, 1 + s_f)

        # Apply transformations
        # TODO: check why sometimes it clips
        translated_logo: PILImage = noisy_logo.transform(noisy_logo.size, Image.Transform.AFFINE,
                                                         (1, 0, tx, 0, 1, ty))
        # TODO: check function, what is expand?
        # TODO: check scale factor
        rotated_and_scaled_logo: PILImage = translated_logo.rotate(angle, expand=True).resize(
            (int(noisy_logo.width * scale_factor), int(noisy_logo.height * scale_factor)))

        # Final position of the center points after transformations
        x_tl_after: float = x_tl + tx
        y_tl_after: float = y_tl + ty

        # place the transformed logo onto the canvas
        # TODO: Is there a way to do it float?
        background.paste(rotated_and_scaled_logo, (int(x_tl_after + c_params.spacing // 2),
                                                   int(y_tl_after + c_params.spacing // 2)),
                         rotated_and_scaled_logo)

        # Store the center points before and after translation
        center_points_after.append((x_tl_after + rotated_and_scaled_logo.width // 2,
                                    y_tl_after + rotated_and_scaled_logo.height // 2))
    # ------------------------------------------------------------------------
    centers_before = pd.DataFrame(center_points_before, columns=["x", "y"])
    centers_after = pd.DataFrame(center_points_after, columns=["x", "y"])
    centers_before.to_csv(base_output_path / 'centers_before.csv', index=False)
    centers_after.to_csv(base_output_path / 'centers_after.cvs', index=False)

    background.save(base_output_path / 'canvas_after.png')


def main():
    base_path: Path = Path('data')
    img_name: str = 'instagram_logo.png'
    canvas_params = CanvasParameters(base_path / 'input' / 'parameters.yaml')
    run(main_folder=base_path, input_img_name=img_name, c_params=canvas_params)


if __name__ == "__main__":
    main()
