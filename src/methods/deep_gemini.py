import cv2
import numpy as np
import os
from imutils.object_detection import non_max_suppression

# Note: This script requires opencv-contrib-python and imutils.
# Install with:
# pip install opencv-contrib-python
# pip install imutils

# ==============================================================================
# ---  PARAMETERS TO TUNE  ---
# ==============================================================================

# --- File Paths ---
TEMPLATE_PATH = 'data_synth/golang/initial_template.png'
SCENE_PATH = 'data_synth/golang/material.bmp'
OUTPUT_IMAGE_PATH = 'data_synth/golang/result_centers.png'
OUTPUT_CENTERS_PATH = 'data_synth/golang/centers_found.txt'  # <<< New output file name

# --- SIFT & Homography Parameters ---
MIN_MATCH_COUNT = 10
LOWE_RATIO = 0.7
RANSAC_REPROJ_THRESHOLD = 5.0
BORDER_SIZE = 30

# --- Template Matching Parameters (Stage 1) ---
TEMPLATE_MATCHING_THRESHOLD = 0.7
NMS_OVERLAP_THRESHOLD = 0.3

# --- Multi-scale Search Parameters ---
MIN_SCALE = 0.5
MAX_SCALE = 2.0
NUM_SCALES = 25


# ==============================================================================
# ---  HELPER FUNCTIONS  ---
# ==============================================================================

def add_border(image, border_size=BORDER_SIZE):
    """Adds a constant black border to an image."""
    return cv2.copyMakeBorder(
        image,
        top=border_size, bottom=border_size, left=border_size, right=border_size,
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )


def detect_and_match_features(template, scene, mask=None):
    """Detects SIFT features, matches them, and returns good matches."""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(scene, mask)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return [], None, None

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(np.float32(des1), np.float32(des2), k=2)

    good_matches = []
    for m_n in matches:
        if len(m_n) == 2 and m_n[0].distance < LOWE_RATIO * m_n[1].distance:
            good_matches.append(m_n[0])
    return good_matches, kp1, kp2


def find_aligned_bounding_box(good_matches, kp1, kp2, template):
    """
    Finds the homography and returns the top-left and bottom-right points
    of the enclosing axis-aligned rectangle.
    """
    if len(good_matches) < MIN_MATCH_COUNT:
        return False, None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)

    if M is None:
        return False, None, None

    # Get the original corners of the template content
    h, w = template.shape[:2]
    h_orig, w_orig = h - 2 * BORDER_SIZE, w - 2 * BORDER_SIZE
    pts = np.float32([
        [BORDER_SIZE, BORDER_SIZE], [BORDER_SIZE, BORDER_SIZE + h_orig],
        [BORDER_SIZE + w_orig, BORDER_SIZE + h_orig], [BORDER_SIZE + w_orig, BORDER_SIZE]
    ]).reshape(-1, 1, 2)

    skewed_corners = cv2.perspectiveTransform(pts, M)

    # Calculate the axis-aligned bounding box
    x_min, y_min = np.int32(skewed_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(skewed_corners.max(axis=0).ravel())

    return True, (x_min, y_min), (x_max, y_max)


# ==============================================================================
# ---  MAIN EXECUTION  ---
# ==============================================================================

def hybrid_object_detection():
    """Runs the hybrid detection method and saves the center points of detected objects."""
    print("Executing Hybrid Template Matching + SIFT Verification...")

    template_orig = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    scene_img = cv2.imread(SCENE_PATH)
    if template_orig is None or scene_img is None:
        print(f"Error: Could not read images. Check paths: '{TEMPLATE_PATH}', '{SCENE_PATH}'")
        return

    scene_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
    template_bordered = add_border(template_orig)
    tH, tW = template_orig.shape[:2]

    # --- Stage 1: Find Candidate ROIs ---
    print("Stage 1: Finding candidate regions...")
    found_rois = []
    # This part is computationally intensive, a more C++ based implementation might be faster
    for scale in np.linspace(MIN_SCALE, MAX_SCALE, NUM_SCALES)[::-1]:
        w_resized, h_resized = int(scene_gray.shape[1] * scale), int(scene_gray.shape[0] * scale)
        if h_resized < tH or w_resized < tW: continue

        resized = cv2.resize(scene_gray, (w_resized, h_resized))
        ratio = scene_gray.shape[1] / float(resized.shape[1])
        result = cv2.matchTemplate(resized, template_orig, cv2.TM_CCOEFF_NORMED)
        y_pts, x_pts = np.where(result >= TEMPLATE_MATCHING_THRESHOLD)

        boxes = [(int(x * r), int(y * r), int((x + tW) * r), int((y + tH) * r)) for x, y, r in
                 zip(x_pts, y_pts, [ratio] * len(x_pts))]
        found_rois.extend(boxes)

    picked_boxes = non_max_suppression(np.array(found_rois), probs=None, overlapThresh=NMS_OVERLAP_THRESHOLD)
    print(f"Found {len(picked_boxes)} unique candidate regions after NMS.")

    # --- Stage 2: SIFT Verification & Center Calculation ---
    print("Stage 2: Verifying regions and calculating center points...")
    all_detected_centers = []

    for (startX, startY, endX, endY) in picked_boxes:
        mask = np.zeros(scene_gray.shape, dtype="uint8")
        cv2.rectangle(mask, (startX, startY), (endX, endY), 255, -1)

        good_matches, kp1, kp2 = detect_and_match_features(template_bordered, scene_gray, mask=mask)

        found, pt1, pt2 = find_aligned_bounding_box(good_matches, kp1, kp2, template_bordered)

        if found:
            # *** NEW: Calculate the center from the bounding box corners ***
            center_x = (pt1[0] + pt2[0]) / 2.0
            center_y = (pt1[1] + pt2[1]) / 2.0
            all_detected_centers.append((center_x, center_y))

            # Draw the bounding box
            cv2.rectangle(scene_img, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
            # Draw a circle at the center for visualization
            cv2.circle(scene_img, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

    print(f"\nVerification complete. Found {len(all_detected_centers)} object(s).")

    # --- Save Results ---
    cv2.imwrite(OUTPUT_IMAGE_PATH, scene_img)
    print(f"Result image saved to '{OUTPUT_IMAGE_PATH}'")

    # *** NEW: Save center points in the specified format ***
    with open(OUTPUT_CENTERS_PATH, 'w') as f:
        f.write('x,y\n')  # Write the header
        for (cx, cy) in all_detected_centers:
            f.write(f'{cx},{cy}\n')  # Write each center on a new line

    print(f"Center coordinates saved to '{OUTPUT_CENTERS_PATH}'")

    cv2.imshow("Verified Detections (Centers)", scene_img)
    print("\nPress any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    hybrid_object_detection()