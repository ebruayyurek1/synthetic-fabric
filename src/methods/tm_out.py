import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ImageEnhance, Image
from mtm import matchTemplates
from mtm.detection import plotDetections
from scipy.ndimage import rotate, zoom
from tqdm import tqdm

def write_bboxes(listDetections, name_output):
    xywhs = [bb.xywh for bb in listDetections]
    bbs = [(x,y, x+w, y+h) for x, y, w, h in xywhs]
    df = pd.DataFrame(bbs, columns=["x_min", "y_min", "x_max", "y_max"])
    df.to_csv(f"out/{name_output}_boxes.csv", index=False)

def write_masks(image, listDetections, name_output):
    xywhs = [bb.xywh for bb in listDetections]
    bbs = [(x,y, x+w, y+h) for x, y, w, h in xywhs]
    whole_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)  # Initialize mask
    for i, (x_min, y_min, x_max, y_max) in enumerate(bbs):
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)  # Initialize mask
        mask[y_min:y_max, x_min:x_max] = 255  # Fill with white pixels
        whole_mask[y_min:y_max, x_min:x_max] = 255  # Fill with white pixels
        Path(f"out/{name_output}_masks").mkdir(exist_ok=True)
        cv2.imwrite(f'out/{name_output}_masks/{i}.png', mask)  # Save the mask
    cv2.imwrite(f'out/{name_output}_masks/all.png', whole_mask)  # Save the mask

def basic_matching(image: np.ndarray, template: np.ndarray,
                   score_threshold: float = 0.5,
                   num_expected_templates: int = float("inf"), name_output: str = "default"):
    listTemplate = [template]
    # Template matching + NMS
    listDetections = matchTemplates(image,
                                    listTemplate,
                                    scoreThreshold=score_threshold,
                                    maxOverlap=0,
                                    nObjects=num_expected_templates)

    plotDetections(image, listDetections)
    write_bboxes(listDetections, name_output)
    write_masks(image, listDetections, name_output)
    plt.savefig(f"out/{name_output}.png")


def augmented_templates_matching(image: np.ndarray, template: np.ndarray,
                                 score_threshold: float = 0.5,
                                 num_expected_templates: int = float("inf"), name_output: str = "default"):
    # --- Augmentation ---
    # Rotation
    all_templates = [template]
    rotated_templates = []
    for i, angle in enumerate([-3, -2, -1, 1, 2, 3]):
        rotated = rotate(template, angle, reshape=False)
        rotated_templates.append(rotated)

    all_templates.extend(rotated_templates)
    # Zoom
    zoomed_templates = []
    for rot_template in rotated_templates:
        zoomed_templates.extend([zoom(rot_template, zoom_value) for zoom_value in [0.95, 0.975, 1.025, 1.05]])
        # We could also do some flipping with np.fliplr, flipud
    all_templates.extend(zoomed_templates)

    # Contrast
    contrastual_logos = []
    for tmpl in all_templates:
        enhancer = ImageEnhance.Contrast((Image.fromarray(tmpl)))
        for value in [0.9, 1.1]:
            contrasted_logo = enhancer.enhance(value)

            contrastual_logos.append(np.asarray(contrasted_logo))
    all_templates.extend(contrastual_logos)

    listDetections = matchTemplates(image,
                                    all_templates,
                                    scoreThreshold=score_threshold,
                                    maxOverlap=0,
                                    nObjects=num_expected_templates)
    plotDetections(image, listDetections)
    write_bboxes(listDetections, name_output)
    write_masks(image, listDetections, name_output)
    plt.savefig(f"out/{name_output}.png")


def main():
    files = ["golang", "java", "unive", "rust", "python"]
    folder = Path("data_synth")
    Path("out").mkdir(exist_ok=True)
    score_threshold = 0.35
    for FILE in tqdm(files, desc="Template matching ..."):
        image = cv2.imread(str(folder / f"{FILE}/material.bmp"), cv2.IMREAD_GRAYSCALE)
        template = cv2.imread(str(folder / f"{FILE}/initial_template.png"), cv2.IMREAD_GRAYSCALE)
        with open(folder / f"{FILE}/centers.csv") as f:
            num_instances = sum(1 for _ in f) - 1
        start = time.perf_counter()
        basic_matching(image, template,
                       num_expected_templates=num_instances,
                       score_threshold=score_threshold,
                       name_output=f"basic_{FILE}")
        print(f"End of basic TM for {FILE}: {time.perf_counter() - start:.4f}")
        start = time.perf_counter()
        augmented_templates_matching(image, template,
                                     num_expected_templates=num_instances,
                                     score_threshold=score_threshold,
                                     name_output=f"augmented_{FILE}")
        print(f"End of advanced TM for {FILE}: {time.perf_counter() - start:.4f}")


if __name__ == '__main__':
    main()
