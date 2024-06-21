from PIL import ImageDraw
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

from src.data_generation.core.CanvasParameters import CanvasParameters
from src.data_generation.core.synt_data_gen import make_background, generate_coordinates, add_centered_gaussian_noise, \
    change_contrast_in_bands, add_centered_gaussian_blur


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
    centers_before: list = generate_coordinates(original_canvas.width, original_canvas.height,
                                                c_params.cols, c_params.rows, c_params.skew)

    for x, y in centers_before:
        # Put in the middle of spacing
        x_tl_w_space, y_tl_w_space = (round(x - cell_width/2), round(y - cell_height/2))
        # box: 2-tuple giving the upper left corner
        original_canvas.paste(logo, (x_tl_w_space, y_tl_w_space), logo)
        # save (floating point) centers
        center_points_before.append((x_tl_w_space + logo.width / 2, y_tl_w_space + logo.height / 2))

    # Save the unaltered canvas
    unaltered_canvas_path: Path = logo_output_path / 'canvas_before.png'
    unaltered_canvas_path.parent.mkdir(exist_ok=True, parents=True)
    # original_canvas.save(unaltered_canvas_path)
    # ------------------------------------------------------------------------
    # Generate random transformations
    # --- Translation ---
    t_f: float = c_params.translation_factor
    t_xs = np.random.uniform(-t_f * cell_width, t_f * cell_width, len(centers_before))
    t_ys = np.random.uniform(-t_f * cell_height, t_f * cell_height, len(centers_before))
    # --- Rotation ---
    r_f: float = c_params.rotation_factor
    angles: np.ndarray = np.random.uniform(-r_f * 360, r_f * 360, len(centers_before))
    # --- Scale ---
    s_f: float = c_params.scale_factor
    scale_factors: np.ndarray = np.random.uniform(1 - s_f, 1 + s_f, len(centers_before))
    # --- Probabilities ---
    p_t, p_r, p_s, p_n, p_g = np.random.uniform(0, 1, 5)
    # Generate random Gaussian noise amounts
    noise_amounts: np.ndarray[float] = np.random.uniform(c_params.min_noise, c_params.max_noise,
                                                         len(centers_before))
    for idx, (x, y) in enumerate(centers_before):
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
            x_tl_after: float = x + t_xs[idx]
            y_tl_after: float = y + t_ys[idx]
        else:
            # Additive probability to ensure it happens
            p_t += np.random.uniform(0, 1)
            x_tl_after: float = x
            y_tl_after: float = y
        # NOTE: Is there a way to do it float?
        # NOTE: Not really...
        # NOTE: We can do an arbitrary resize
        # Putting logos onto the canvas with random transformations and random noise
        x = round(x_tl_after + c_params.spacing / 2)
        y = round(y_tl_after + c_params.spacing / 2)

        enhancer = ImageEnhance.Contrast(scaled_logo)
        scaled_logo = enhancer.enhance(np.random.uniform(c_params.logo_contrast_min, c_params.logo_contrast_max))

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
            p_n += np.random.uniform(0, 1)  #

        # Gaussian noise to the logo
        # --- BLUR ---
        if p_g >= 1 - c_params.transform_prob:
            # Reset probability to a random value
            p_g = np.random.uniform(0, 1)
            background = add_centered_gaussian_blur(background, *center_points_after[-1], shape=scaled_logo.size)
        else:
            # Additive probability to ensure it happens
            p_g += np.random.uniform(0, 1)

    background = change_contrast_in_bands(background, c_params)
    # ------------------------------------------------------------------------
    # Save center coordinates
    # centers_before = pd.DataFrame(center_points_before, columns=["x", "y"])
    centers_after = pd.DataFrame(center_points_after, columns=["x", "y"])
    centers_after.to_csv(logo_output_path / f'{c_params.bg_texture.stem}_{image_path.stem}_centers.csv', index=False)
    # centers_after.to_csv(logo_output_path / 'centers_after.csv', index=False)
    #
    background.save(logo_output_path / f'{c_params.bg_texture.stem}_{image_path.stem}_image.png')

    # DEBUG
    draw = ImageDraw.Draw(background)
    for (x, y) in center_points_after:
        draw.rectangle((x - 10, y - 10, x + 10, y + 10), outline='red', fill="white")
        draw.point((x, y), fill="red")
    # write to stdout
    background.save(logo_output_path / f'{c_params.bg_texture.stem}_{image_path.stem}_debug.png')


def main():
    base_path: Path = Path('data')
    # img_names: list[str] = ['tucano.png', 'instagram_logo.png', 'starbucks_logo.png', 'CocaCola_logo.png']
    bgs = [Path("data/textures/new/burlap.jpg")] # list(Path("data/textures/new").glob('*.jpg'))
    img_names: list[str] = ['python.png']# ['rust.png', 'unive.png', 'python.png', 'golang.png']
    for bg in bgs:
        times = []
        for img_name in img_names:
            canvas_params = CanvasParameters(base_path / 'input' / 'parameters.yaml')
            canvas_params.bg_texture = bg
            canvas_params.rows = random.randint(5, 10)
            canvas_params.cols = random.randint(5, 10)
            canvas_params.spacing = random.randint(50, 300)
            canvas_params.interval_skew_fraction = random.uniform(0.0, 0.5)

            start = time.perf_counter()
            run(main_folder=base_path,
                input_img_name=img_name,
                c_params=canvas_params)
            times.append(time.perf_counter() - start)
            print(f"{img_name}", sep=" ")
        print(f'Average time elapsed for "{bg.stem}": {np.average(times):.2f}s')


if __name__ == "__main__":
    main()
