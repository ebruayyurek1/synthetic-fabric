from pathlib import Path

import cv2
import numpy as np
import pandas as pd
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
        clean_bg = cv2.imread(str(path / f"{filename}1.bmp"))
        img = clean_bg.copy()
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

        boxes = merge(clean_bg, boxes, 1 if filename == "java" else 0)

        centers = [((x1 + x2) / 2, (y1 + y2) / 2) for (x1, y1), (x2, y2) in boxes]

        df = pd.DataFrame(centers, columns=["x", "y"])
        df.to_csv(f"out/yolo_{filename}_centers.csv", index=False)


# --- MERGE BOXES ---
def overlap(source, target):
    # unpack points
    tl1, br1 = source
    tl2, br2 = target

    # checks
    # if the x of the source is to the right of the br of the target, can-t overlap
    # Also if tl of the target is to the right of the br of target
    if tl1[0] >= br2[0] or tl2[0] >= br1[0]:
        return False
    # Same reasoning but for the y of source tl is below of br
    # Or tl of target is below br of source
    if tl1[1] >= br2[1] or tl2[1] >= br1[1]:
        return False
    return True

    # returns all overlapping boxes


def getAllOverlaps(boxes, bounds, index):
    overlaps = []
    for a in range(len(boxes)):
        if a != index:
            if overlap(bounds, boxes[a]):
                overlaps.append(a)
    return overlaps


def merge(clean_bg: np.ndarray, boxes: list, merge_margin: float = 0) -> list:
    image = clean_bg.copy()
    plt.figure(figsize=(15, 15))
    # loop through boxes
    boxes = [[[x1, y1], [x2, y2]] for (x1, y1, x2, y2) in boxes]
    # loop through boxes
    index = len(boxes) - 1
    while index >= 0:
        # grab current box
        curr = boxes[index]

        # add margin
        tl = curr[0]
        br = curr[1]
        tl[0] -= merge_margin
        tl[1] -= merge_margin
        br[0] += merge_margin
        br[1] += merge_margin

        # get matching boxes
        overlaps = getAllOverlaps(boxes, [tl, br], index)

        # check if empty
        if len(overlaps) > 0:
            # combine boxes
            # convert to a contour
            con = []
            overlaps.append(index)
            for ind in overlaps:
                tl, br = boxes[ind][0], boxes[ind][1]
                con.append([tl])
                con.append([br])
            con = np.array(con)
            # get bounding rect
            x, y, w, h = cv2.boundingRect(con)

            # stop growing
            w -= 1
            h -= 1
            merged = [[x, y], [x + w, y + h]]

            # remove boxes from list
            overlaps.sort(reverse=True)
            for ind in overlaps:
                del boxes[ind]
            boxes.append(merged)
        # increment
        if len(overlaps) == 0:
            index -= 1
        else:
            index = len(boxes) - 1

    for box in boxes:
        (x_min, y_min), (x_max, y_max) = box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    plt.imshow(image)
    plt.show()

    return boxes


if __name__ == '__main__':
    main()
