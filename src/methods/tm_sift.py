import time
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mtm import matchTemplates
from mtm.detection import plotDetections
from PIL import Image, ImageEnhance
from scipy.ndimage import rotate, zoom
from scipy.spatial.distance import cdist
from tqdm import tqdm


def plot_sift_detections(image, detections, name_output):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image, cmap='gray')
    for det in detections:
        x, y, w, h = det.xywh
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"out/{name_output}.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def basic_matching(image: np.ndarray, template: np.ndarray,
                   score_threshold: float = 0.5,
                   num_expected_templates: int = float("inf"), name_output: str = "default"):
    listTemplate = [template]
    listDetections = matchTemplates(image,
                                    listTemplate,
                                    scoreThreshold=score_threshold,
                                    maxOverlap=0,
                                    nObjects=num_expected_templates)
    write_centers(listDetections, name_output)
    plotDetections(image, listDetections)
    plt.savefig(f"out/{name_output}.png")
    plt.close()


def augmented_templates_matching(image: np.ndarray, template: np.ndarray,
                                 score_threshold: float = 0.5,
                                 num_expected_templates: int = float("inf"), name_output: str = "default"):
    all_templates = [template]
    rotated_templates = [rotate(template, angle, reshape=False) for angle in [-3, -2, -1, 1, 2, 3]]
    all_templates.extend(rotated_templates)
    zoomed_templates = []
    for rot_template in rotated_templates:
        zoomed_templates.extend([zoom(rot_template, zoom_value) for zoom_value in [0.95, 0.975, 1.025, 1.05]])
    all_templates.extend(zoomed_templates)
    contrastual_logos = []
    for tmpl in all_templates:
        enhancer = ImageEnhance.Contrast(Image.fromarray(tmpl))
        for value in [0.9, 1.1]:
            contrastual_logos.append(np.asarray(enhancer.enhance(value)))
    all_templates.extend(contrastual_logos)
    listDetections = matchTemplates(image,
                                    all_templates,
                                    scoreThreshold=score_threshold,
                                    maxOverlap=0,
                                    nObjects=num_expected_templates)
    plotDetections(image, listDetections)
    write_centers(listDetections, name_output)
    plt.savefig(f"out/{name_output}.png")
    plt.close()


def _feature_alignment_core(image: np.ndarray, template: np.ndarray, detector_name: str,
                            score_threshold: float, num_expected_templates: int, name_output: str):
    """
    Core function for hybrid matching using a specified feature detector (SIFT, ORB, BRISK, FREAK).
    """
    # --- Parameters ---
    MIN_MATCH_COUNT = 8
    LOWE_RATIO = 0.7
    RANSAC_REPROJ_THRESHOLD = 4.0

    class Detection:
        def __init__(self, xywh):
            self.xywh = tuple(map(int, xywh))

    # --- Step 1: Initialize correct feature detector and matcher ---
    is_binary_descriptor = False
    if detector_name == 'sift':
        detector = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    elif detector_name in ['orb', 'brisk']:
        is_binary_descriptor = True
        if detector_name == 'orb': detector = cv2.ORB_create(nfeatures=1000)
        if detector_name == 'brisk': detector = cv2.BRISK_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif detector_name == 'freak':
        is_binary_descriptor = True
        detector = cv2.SIFT_create()
        extractor = cv2.xfeatures2d.FREAK_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        raise ValueError("Unsupported detector specified.")

    # --- Step 2: Find initial candidates with template matching ---
    initial_detections = matchTemplates(image, [template], scoreThreshold=score_threshold, maxOverlap=0.1, nObjects=num_expected_templates)
    candidate_boxes = [(int(det.xywh[0]), int(det.xywh[1]), int(det.xywh[2]), int(det.xywh[3])) for det in initial_detections]

    # --- Step 3: Refine each candidate's bounding box ---
    final_detections = []
    tH, tW = template.shape[:2]
    for (startX, startY, w, h) in candidate_boxes:
        roi = image[startY:startY + h, startX:startX + w]
        if roi.size == 0: continue

        padded_roi = np.zeros((tH, tW), dtype=np.uint8)
        y_offset, x_offset = (tH - h) // 2, (tW - w) // 2
        padded_roi[y_offset:y_offset + h, x_offset:x_offset + w] = roi

        # Detect and match features
        if detector_name == 'freak':
            kp1 = detector.detect(template, None)
            kp1, des1 = extractor.compute(template, kp1)
            kp2 = detector.detect(padded_roi, None)
            kp2, des2 = extractor.compute(padded_roi, kp2)
        else:
            kp1, des1 = detector.detectAndCompute(template, None)
            kp2, des2 = detector.detectAndCompute(padded_roi, None)

        if des1 is None or des2 is None:
            final_detections.append(Detection(xywh=(startX, startY, w, h)))
            continue

        # CORRECTED MATCHING LOGIC
        if is_binary_descriptor:
            matches = matcher.match(des1, des2)
            good_matches = sorted(matches, key=lambda x: x.distance)[:int(len(matches) * 0.2)]
        else:  # SIFT
            matches = matcher.knnMatch(des1, des2, k=2)
            good_matches = []
            # Apply Lowe's ratio test safely
            for match_pair in matches:
                if len(match_pair) == 2:  # Ensure we have a pair of matches
                    m, n = match_pair
                    if m.distance < LOWE_RATIO * n.distance:
                        good_matches.append(m)

        found = False
        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)

            if M is not None:
                pts = np.float32([[0, 0], [0, tH - 1], [tW - 1, tH - 1], [tW - 1, 0]]).reshape(-1, 1, 2)
                skewed_corners = cv2.perspectiveTransform(pts, M)
                x_min, y_min = np.int32(skewed_corners.min(axis=0).ravel())
                x_max, y_max = np.int32(skewed_corners.max(axis=0).ravel())

                x_global, y_global = x_min - x_offset + startX, y_min - y_offset + startY
                w_aligned, h_aligned = x_max - x_min, y_max - y_min

                if w_aligned > 0 and h_aligned > 0:
                    final_detections.append(Detection(xywh=(x_global, y_global, w_aligned, h_aligned)))
                    found = True

        if not found:
            final_detections.append(Detection(xywh=(startX, startY, w, h)))

    # --- Step 4: Write outputs ---
    plot_sift_detections(image, final_detections, name_output)
    write_centers(final_detections, name_output)

# --- Public-facing wrapper functions ---
def sift_alignment_matching(image: np.ndarray, template: np.ndarray, score_threshold: float,
                            num_expected_templates: int, name_output: str):
    _feature_alignment_core(image, template, 'sift', score_threshold, num_expected_templates, name_output)


def orb_alignment_matching(image: np.ndarray, template: np.ndarray, score_threshold: float, num_expected_templates: int,
                           name_output: str):
    _feature_alignment_core(image, template, 'orb', score_threshold, num_expected_templates, name_output)


def brisk_alignment_matching(image: np.ndarray, template: np.ndarray, score_threshold: float,
                             num_expected_templates: int, name_output: str):
    _feature_alignment_core(image, template, 'brisk', score_threshold, num_expected_templates, name_output)


def freak_alignment_matching(image: np.ndarray, template: np.ndarray, score_threshold: float,
                             num_expected_templates: int, name_output: str):
    _feature_alignment_core(image, template, 'freak', score_threshold, num_expected_templates, name_output)


def write_centers(listDetections, name_output):
    xywhs = [bb.xywh for bb in listDetections]
    centers = [(x + w / 2, y + h / 2) for x, y, w, h in xywhs]
    df = pd.DataFrame(centers, columns=["x", "y"])
    df.to_csv(f"out/{name_output}_centers.csv", index=False)


def _match_true_with_predicted(true_centers: np.ndarray, predicted_centers: np.ndarray, matching_threshold_px: float) \
        -> tuple[list[tuple[Optional[int], Optional[int]]], np.ndarray]:
    if len(predicted_centers) == 0 or len(true_centers) == 0:
        return [], np.array([])
    pairwise_distances = cdist(true_centers, predicted_centers, "euclidean")
    matches, pred_matched_indexes = [], set()
    for i in range(len(true_centers)):
        below_threshold_indexes = np.argwhere(pairwise_distances[i] < matching_threshold_px).squeeze(1)
        if len(below_threshold_indexes) == 1 and below_threshold_indexes[0] not in pred_matched_indexes:
            matched_index = int(below_threshold_indexes[0])
            matches.append((i, matched_index))
            pred_matched_indexes.add(matched_index)
        else:
            matches.append((i, None))
    for j in range(len(predicted_centers)):
        if j not in pred_matched_indexes:
            matches.append((None, j))
    return matches, pairwise_distances


def compute_metrics(true_centers: np.ndarray, predicted_centers: np.ndarray,
                    matching_threshold_px: float = 5.0) -> dict:
    matches, pairwise_distances = _match_true_with_predicted(true_centers, predicted_centers, matching_threshold_px)
    error = np.array([pairwise_distances[i, j] for i, j in matches if i is not None and j is not None])

    if not len(error):
        return {"mae": -1, "rmse": -1, "recall": 0, "h_mean": 0}

    mae = float(error.mean())
    rmse = float(np.sqrt(np.square(error).mean()))
    recall = len(error) / len(true_centers) if len(true_centers) > 0 else 0
    precision_rmse = 1 - rmse / matching_threshold_px
    h_mean = 2 * (recall * precision_rmse) / (recall + precision_rmse) if (recall + precision_rmse) > 0 else 0

    return {"mae": mae, "rmse": rmse, "recall": recall, "h_mean": h_mean}


def run_experiments(n_runs: int = 5):

    files = ["golang", "java", "unive", "rust", "python"]
    folder = Path("data_synth")
    score_threshold = 0.35

    methods = {
        "basic": basic_matching,
        # "augmented": augmented_templates_matching,
        "sift": sift_alignment_matching,
        "orb": orb_alignment_matching,
        "brisk": brisk_alignment_matching,
        "freak": freak_alignment_matching,
    }

    all_runs_data = {method_name: [] for method_name in methods}

    for run_idx in range(n_runs):
        print(f"\n{'=' * 20} RUN {run_idx + 1}/{n_runs} {'=' * 20}")

        run_metrics = {method_name: [] for method_name in methods}
        run_timings = {method_name: [] for method_name in methods}

        for FILE in tqdm(files, desc=f"Processing images for run {run_idx + 1}"):
            Path("out").mkdir(exist_ok=True)
            image = cv2.imread(str(folder / f"{FILE}/material.bmp"), cv2.IMREAD_GRAYSCALE)
            template = cv2.imread(str(folder / f"{FILE}/initial_template.png"), cv2.IMREAD_GRAYSCALE)
            true_centers = pd.read_csv(folder / f"{FILE}/centers.csv").to_numpy()
            num_instances = len(true_centers)

            for method_name, func in methods.items():
                output_name = f"{method_name}_{FILE}_run{run_idx + 1}"

                start_time = time.perf_counter()
                func(image.copy(), template,
                     num_expected_templates=num_instances,
                     score_threshold=score_threshold,
                     name_output=output_name)
                end_time = time.perf_counter()
                run_timings[method_name].append(end_time - start_time)

                pred_centers_path = Path(f"out/{output_name}_centers.csv")
                if pred_centers_path.exists():
                    predicted_centers = pd.read_csv(pred_centers_path).to_numpy()
                    metrics = compute_metrics(true_centers, predicted_centers)
                    run_metrics[method_name].append(metrics)

        # Aggregate metrics and total time for the current run
        for method_name in methods:
            df_run = pd.DataFrame(run_metrics[method_name])
            avg_run_metrics = df_run.mean().to_dict()
            avg_run_metrics['time'] = np.sum(run_timings[method_name])
            all_runs_data[method_name].append(avg_run_metrics)

    # --- Final Reporting ---
    print(f"\n{'=' * 25} FINAL REPORT (Avg over {n_runs} runs) {'=' * 25}")
    for method_name in methods:
        print(f"\n--- Method: {method_name.capitalize()} ---")
        df_final = pd.DataFrame(all_runs_data[method_name])

        means = df_final.mean()
        stds = df_final.std()

        for metric in means.index:
            print(f"  {metric.upper():<8}: {means[metric]:.4f} ± {stds[metric]:.4f}")


if __name__ == '__main__':
    run_experiments()