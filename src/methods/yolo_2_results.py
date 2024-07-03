from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def main():
    path = Path("data_synth/yolo_preds")
    files = ["golang", "java", "python", "rust", "unive"]
    for filename in files:
        img = cv2.imread(str(path / f"{filename}1.bmp"))
        image_height, image_width = img.shape[:-1]
        with open(str(path / f"{filename}.txt"), "rb") as f:
            results = f.readlines()

        boxes = []
        plt.figure(figsize=(15, 15))
        # Parse each line
        for line in results:
            parts = line.strip().split()  # Split by whitespace
            if len(parts) == 5:  # Ensure all expected values are present
                class_label, x_norm, y_norm, width_norm, height_norm = map(float, parts)
                # 1 = line; skip
                if class_label != 1:
                    xyxy = xywh2xyxy(np.array([x_norm, y_norm, width_norm, height_norm]))
                    x1_pixel, y1_pixel, x2_pixel, y2_pixel = xyxy
                    # Convert normalized coordinates to pixel values (assuming image dimensions)

                    x1_pixel = int(x1_pixel * image_width)
                    y1_pixel = int(y1_pixel * image_height)
                    x2_pixel = int(x2_pixel * image_width)
                    y2_pixel = int(y2_pixel * image_height)
                    boxes.append((x1_pixel, y1_pixel, x2_pixel, y2_pixel))

        for box in boxes:
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        plt.imshow(img)
        plt.show()


# --- MERGE BOXES ---
# def overlap(source, target):
#     # unpack points
#     tl1, br1 = source
#     tl2, br2 = target
#
#     # checks
#     if tl1[0] >= br2[0] or tl2[0] >= br1[0]:
#         return False
#     if tl1[1] >= br2[1] or tl2[1] >= br1[1]:
#         return False
#     return True
#
#     # returns all overlapping boxes
#
#
# def getAllOverlaps(boxes, bounds, index):
#     overlaps = []
#     for a in range(len(boxes)):
#         if a != index:
#             if overlap(bounds, boxes[a]):
#                 overlaps.append(a)
#     return overlaps



def merge(image: np.ndarray, boxes: list, threshold: float = 0.1) -> list:
    """
    Merges overlapping or very close bounding boxes.

    Args:
        image: Input image (not used for merging, only for visualization).
        boxes: List of bounding boxes (each box as [x_top_left, y_top_left, x_bottom_right, y_bottom_right]).
        threshold: Distance threshold for merging (adjust as needed).

    Returns:
        List of merged bounding boxes.
    """
    merged_boxes = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        merged = False

        for j in range(len(merged_boxes)):
            mx1, my1, mx2, my2 = merged_boxes[j]
            distance = abs(mx1 - x2)  # Horizontal distance (you can adjust this)

            if distance < threshold:
                # Merge the boxes
                new_x1 = min(x1, mx1)
                new_x2 = max(x2, mx2)
                new_y1 = min(y1, my1)
                new_y2 = max(y2, my2)
                merged_boxes[j] = [new_x1, new_y1, new_x2, new_y2]
                merged = True
                break

        if not merged:
            merged_boxes.append([x1, y1, x2, y2])

    # Visualize merged boxes (optional)
    for rect in merged_boxes:
        cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

    # Save or display the modified image (optional)
    cv2.imwrite("merged_boxes.png", image)

    return merged_boxes


if __name__ == '__main__':
    main()
