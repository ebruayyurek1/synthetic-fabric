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


# Credit to:
# Thomas, L.S.V., Gehrig, J.
# Multi-template matching: a versatile tool for object-localization in microscopy images.
# BMC Bioinformatics 21, 44 (2020). https://doi.org/10.1186/s12859-020-3363-7
# mtm-skimage-shapely python package online tutorial - Tutorial 2: Template Augmentation
# Laurent Thomas - 2021
# https://github.com/multi-template-matching/mtm-python-oop/blob/master/tutorials/Tutorial2-Template_Augmentation.ipynb

def write_centers(listDetections, name_output):
    xywhs = [bb.xywh for bb in listDetections]
    centers = [(x + w / 2, y + h / 2) for x, y, w, h in xywhs]
    df = pd.DataFrame(centers, columns=["x", "y"])
    df.to_csv(f"out/{name_output}_centers.csv", index=False)


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

    write_centers(listDetections, name_output)

    plotDetections(image, listDetections)
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

    # --- PLOT ---
    # f, axarr = plt.subplots(round(np.sqrt(len(all_templates)))+1, round(np.sqrt(len(all_templates))), figsize=(20, 20))
    # i, j = 0, 0
    # for tmpl in all_templates:
    #     axarr[i][j].imshow(tmpl, cmap="gray")
    #     i += 1
    #     if i == round(np.sqrt(len(all_templates)))+1:
    #         j += 1
    #         i = 0
    # plt.tight_layout()
    # plt.show()
    # -----------
    # Template matching + NMS
    listDetections = matchTemplates(image,
                                    all_templates,
                                    scoreThreshold=score_threshold,
                                    maxOverlap=0,
                                    nObjects=num_expected_templates)

    plotDetections(image, listDetections)
    write_centers(listDetections, name_output)
    plt.savefig(f"out/{name_output}.png")


def main():
    files = ["golang", "java", "unive", "rust", "python"]
    folder = Path("data_synth")
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
