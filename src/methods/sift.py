# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_cross_keypoints(img, keypoints):
    """ Draw keypoints as crosses, and return the new image with the crosses. """
    img_kp = img.copy()  # Create a copy of img

    plt.figure(figsize=(30, 30))
    img_kp = cv2.drawKeypoints(img_kp, keypoints, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.imshow(img_kp)
    plt.show()


def find_matching_boxes(image, template, detector_method, params):
    # Parameters and their default values
    MAX_MATCHING_OBJECTS = params.get('max_matching_objects', 10)
    SIFT_DISTANCE_THRESHOLD = params.get('SIFT_distance_threshold', 0.3)
    BEST_MATCHES_POINTS = params.get('best_matches_points', 50)

    # "contrastThreshold": 0.01
    # "edgeThreshold": 10
    # "nfeatures": 200
    # nOctaveLayers=3,
    # # (default = 3) Default should be ok
    # contrastThreshold=0.01,
    # # (default = 0.04) Lower = Include kps with lower contrast
    # edgeThreshold=30,
    # # (default = 10) Higher = Include KPS with lower edge response
    # sigma=1.6)
    # Initialize the detector and matcher
    if detector_method == "SIFT":
        detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.03, edgeThreshold=20, sigma=1.5)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # matcher = cv2.BFMatcher()
    elif detector_method == "ORB":
        detector = cv2.ORB_create(fastThreshold=5, edgeThreshold=10)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        raise ValueError("Unsupported detector method")

    # Find keypoints and descriptors for the template
    keypoints2, descriptors2 = detector.detectAndCompute(template, None)

    matched_boxes = []
    matching_img = image.copy()

    for i in range(MAX_MATCHING_OBJECTS):
        # Match descriptors
        keypoints1, descriptors1 = detector.detectAndCompute(matching_img, None)
        draw_cross_keypoints(image, keypoints1)
        # quit()
        if detector_method == "SIFT":
            # Matching strategy for SIFT
            matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
            good_matches = []
            # for m, n in matches:
            #     if m.distance < SIFT_DISTANCE_THRESHOLD * n.distance:
            #         good_matches.append([m])
            good_matches = [m for m, n in matches if m.distance < SIFT_DISTANCE_THRESHOLD * n.distance]
            good_matches = sorted(good_matches, key=lambda x: x.distance)[:BEST_MATCHES_POINTS]

        elif detector_method == "ORB":
            # Matching strategy for ORB
            matches = matcher.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:BEST_MATCHES_POINTS]

        else:
            raise ValueError("Unsupported detector method")

        # Extract location of good matches
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

        # Find homography for drawing the bounding box
        try:
            H, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 2)
        except cv2.error:
            print("No more matching box")
            break

        # Transform the corners of the template to the matching points in the image
        h, w = template.shape[:2]
        corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        matched_boxes.append(transformed_corners)

        # # You can uncomment the following lines to see the matching process
        # # Draw the bounding box
        img1_with_box = matching_img.copy()
        matching_result = cv2.drawMatches(img1_with_box, keypoints1, template, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.polylines(matching_result, [np.int32(transformed_corners)], True, (255, 0, 0), 3, cv2.LINE_AA)
        plt.imshow(matching_result, cmap='gray')
        plt.show()

        # Create a mask and fill the matched area with near neighbors
        mask = np.ones_like(matching_img) * 255
        cv2.fillPoly(mask, [np.int32(transformed_corners)], 0)
        mask = cv2.bitwise_not(mask)
        matching_img = cv2.inpaint(matching_img, mask, 3, cv2.INPAINT_TELEA)

    return matched_boxes


def main():
    # Example usage:
    matching_method = "SIFT"  # "SIFT" or "ORB"

    img1 = cv2.imread("datasets/new_test_imgs/java/image.png", cv2.IMREAD_GRAYSCALE)  # Image
    template = cv2.imread("datasets/new_test_imgs/java/initial_template.png", cv2.IMREAD_GRAYSCALE)  # Template

    params = {
        'max_matching_objects': 150,
        'SIFT_distance_threshold': 0.4,
        'best_matches_points': 50
    }

    # Change to "SIFT" or "ORB" depending on your requirement
    matched_boxes = find_matching_boxes(img1, template, matching_method, params)

    # Draw the bounding boxes on the original image
    for box in matched_boxes:
        cv2.polylines(img1, [np.int32(box)], True, (0, 255, 0), 3, cv2.LINE_AA)

    plt.imshow(img1)
    plt.show()

    # %%


if __name__ == '__main__':
    main()
