from PIL import Image, ImageDraw
import mediapipe as mp
import numpy as np
import cv2  # OpenCV is used for image format conversion
import itertools
import random
from sklearn.cluster import MeanShift


def import_model():
    """
    Initializes and returns the MediaPipe Selfie Segmentation model.
    """
    print("Initializing MediaPipe Selfie Segmentation model...")
    selfie_segmentation = mp.solutions.selfie_segmentation
    model = selfie_segmentation.SelfieSegmentation(model_selection=0)
    print("Model initialized successfully.")
    return model


def run_full_analysis(input_image, model):
    """
    Runs segmentation and perspective analysis, then combines them into an
    aesthetically pleasing compositional template.
    """
    # 1. Get the person mask
    person_mask = image_segmentation(input_image, model)

    # 2. Find the main horizon/tilted line
    horizon_line = find_horizon_line(input_image)

    # 3. Find the single vanishing point and its corresponding perspective lines
    vanishing_point, perspective_lines = find_one_point_perspective(input_image)

    # --- 4. Create a new, visually appealing template ---
    height, width = input_image.size[1], input_image.size[0]

    # Create a transparent RGBA canvas to draw on
    template_np = np.zeros((height, width, 4), dtype=np.uint8)

    # Dynamic, thicker lines
    line_thickness = max(2, int(min(height, width) * 0.005 * 1.7))
    person_outline_thickness = max(2, int(min(height, width) * 0.005 * 1.7))

    # Re-introduce the correct overlap logic from the stable version
    if horizon_line and perspective_lines:

        def get_angle(line_coords):
            x1, y1, x2, y2 = line_coords
            return np.degrees(np.arctan2(y2 - y1, x2 - x1))

        horizon_angle = get_angle(horizon_line)
        for p_line in perspective_lines:
            p_line_angle = get_angle(p_line)
            if abs(horizon_angle - p_line_angle) < 5.0:
                print(
                    "   ...Horizon line overlaps with perspective. Forcing horizontal."
                )
                avg_y = int((horizon_line[1] + horizon_line[3]) / 2)
                horizon_line = [0, avg_y, width, avg_y]
                break

    img_rect = (0, 0, width, height)
    line_color = (255, 255, 255, 204)  # White with 80% opacity

    # Draw compositional lines FIRST
    if perspective_lines and vanishing_point:
        vp_x, vp_y = vanishing_point
        for line in perspective_lines:
            x1, y1, x2, y2 = line
            dist1 = np.sqrt((x1 - vp_x) ** 2 + (y1 - vp_y) ** 2)
            dist2 = np.sqrt((x2 - vp_x) ** 2 + (y2 - vp_y) ** 2)
            direction_point = (x1, y1) if dist1 > dist2 else (x2, y2)
            dx = direction_point[0] - vp_x
            dy = direction_point[1] - vp_y
            factor = 2000
            p_far = (int(vp_x + dx * factor), int(vp_y + dy * factor))
            is_inside, pt1_clipped, pt2_clipped = cv2.clipLine(
                img_rect, vanishing_point, p_far
            )
            if is_inside:
                cv2.line(
                    template_np, pt1_clipped, pt2_clipped, line_color, line_thickness
                )

    if horizon_line:
        x1, y1, x2, y2 = horizon_line
        cv2.line(template_np, (x1, y1), (x2, y2), line_color, line_thickness)

    # --- AESTHETIC CHANGE: Create a super-smooth, connected outline ---
    if person_mask:
        mask_np = np.array(person_mask)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            clean_mask = np.zeros_like(mask_np)
            cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)

            kernel = np.ones((50, 50), np.uint8)
            closed_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

            # --- NEW SUPER-SMOOTHING LOGIC ---
            # 1. Apply a very strong Gaussian blur to "melt" the pixelated edges
            blur_kernel_size = int(min(height, width) * 0.1)
            if blur_kernel_size % 2 == 0:
                blur_kernel_size += 1  # Kernel must be odd
            blurred_mask = cv2.GaussianBlur(
                closed_mask, (blur_kernel_size, blur_kernel_size), 0
            )

            # 2. Re-threshold the blurred image to create a new, sharp, and smooth mask
            _, smooth_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)

            # 3. Find the contour of this new super-smooth mask
            final_contours, _ = cv2.findContours(
                smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Erase lines inside the person's final shape
            eraser_mask = np.zeros_like(mask_np)
            cv2.drawContours(eraser_mask, final_contours, -1, 255, -1)  # -1 to fill
            template_np[eraser_mask == 255] = [0, 0, 0, 0]  # Set to transparent

            # Draw the final smooth outline
            cv2.drawContours(
                template_np,
                final_contours,
                -1,
                (255, 255, 255, 180),
                person_outline_thickness,
            )

    # Vanishing point is no longer drawn.

    output_image = Image.fromarray(template_np)
    return output_image


def image_segmentation(input_image, model):
    """
    Uses MediaPipe Selfie Segmentation to create a mask of the person in the image.
    """
    if model is None:
        return None
    image_np = np.array(input_image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    results = model.process(image_bgr)
    mask = results.segmentation_mask > 0.5
    mask_image_np = (mask * 255).astype(np.uint8)
    output_image = Image.fromarray(mask_image_np).resize(input_image.size)
    return output_image


def find_one_point_perspective(input_image, num_lines_to_draw=4):
    """
    Detects a single dominant vanishing point using the Line Segment Detector (LSD)
    and MeanShift clustering for robust intersection analysis.
    """
    print("   Detecting one-point perspective with LSD and Clustering...")
    image_np = np.array(input_image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(gray)

    if lines is None or len(lines) < 4:
        return None, []

    perspective_lines_candidates = []
    min_line_len = int(min(image_np.shape[:2]) * 0.15)
    for line in lines:
        x1, y1, x2, y2 = map(int, line[0])
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < min_line_len:
            continue

        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angle_tolerance = 15
        if (abs(angle) > angle_tolerance and abs(angle) < (90 - angle_tolerance)) or (
            abs(angle) > (90 + angle_tolerance) and abs(angle) < (180 - angle_tolerance)
        ):
            perspective_lines_candidates.append(line)

    if len(perspective_lines_candidates) < 4:
        return None, []

    intersections = []
    for line1, line2 in itertools.combinations(perspective_lines_candidates, 2):
        x1, y1, x2, y2 = line1[0]
        x3, y3, x4, y4 = line2[0]
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-6:
            continue
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        px = int(x1 + t * (x2 - x1))
        py = int(y1 + t * (y2 - y1))
        img_diag = np.sqrt(image_np.shape[0] ** 2 + image_np.shape[1] ** 2)
        if abs(px) < 2 * img_diag and abs(py) < 2 * img_diag:
            intersections.append((px, py))

    if len(intersections) < 2:
        return None, []

    ms = MeanShift(bandwidth=50, bin_seeding=True)
    ms.fit(intersections)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    unique_labels, counts = np.unique(labels, return_counts=True)

    if len(counts) == 0:
        return None, []

    largest_cluster_idx = np.argmax(counts)
    vanishing_point = tuple(cluster_centers[largest_cluster_idx].astype(int))

    min_inlier_threshold = max(4, int(len(perspective_lines_candidates) * 0.3))
    if counts[largest_cluster_idx] < min_inlier_threshold:
        return None, []

    print(
        f"   ...Vanishing point detected at {vanishing_point} with a cluster of {counts[largest_cluster_idx]} intersections."
    )

    inlier_lines = []
    for line in perspective_lines_candidates:
        x1, y1, x2, y2 = line[0]
        dist = np.abs(
            (y2 - y1) * vanishing_point[0]
            - (x2 - x1) * vanishing_point[1]
            + x2 * y1
            - y2 * x1
        ) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        if dist < 30:
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            inlier_lines.append(
                {"line": (int(x1), int(y1), int(x2), int(y2)), "length": length}
            )

    quadrants = {"tl": [], "tr": [], "bl": [], "br": []}
    vp_x, vp_y = vanishing_point
    for line_data in inlier_lines:
        x1, y1, x2, y2 = line_data["line"]
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        if mid_x < vp_x and mid_y < vp_y:
            quadrants["tl"].append(line_data)
        elif mid_x >= vp_x and mid_y < vp_y:
            quadrants["tr"].append(line_data)
        elif mid_x < vp_x and mid_y >= vp_y:
            quadrants["bl"].append(line_data)
        else:
            quadrants["br"].append(line_data)

    strongest_lines = []
    for quadrant in quadrants.values():
        if quadrant:
            longest_line = max(quadrant, key=lambda x: x["length"])
            strongest_lines.append(longest_line["line"])

    return vanishing_point, strongest_lines


def find_horizon_line(input_image):
    """
    Finds the single most dominant line (even if tilted) by rotating the image's
    gradient map.
    """
    print("   Detecting horizon line...")
    image_np = np.array(input_image)
    height, width, _ = image_np.shape
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    abs_sobel_y = np.uint8(np.absolute(sobel_y))
    best_angle, max_strength, best_row = 0, 0, 0
    for angle in np.arange(-45, 46, 1):
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rotated_sobel = cv2.warpAffine(abs_sobel_y, M, (width, height))
        row_sums = np.sum(rotated_sobel, axis=1)
        current_max = np.max(row_sums)
        if current_max > max_strength:
            max_strength, best_angle, best_row = current_max, angle, np.argmax(row_sums)
    print(f"   ...Horizon line found at angle {best_angle} degrees.")
    if max_strength < (
        np.mean(abs_sobel_y) * width * 0.1
    ):  # Threshold to avoid finding horizon in images without one
        print("   ...Horizon line too weak, discarding.")
        return None

    center_x, center_y = width / 2, height / 2
    rad_angle = np.deg2rad(best_angle)
    orig_y = (best_row - center_y) * np.cos(-rad_angle) + center_y
    orig_x = (best_row - center_y) * np.sin(-rad_angle) + center_x
    x1, y1 = 0, np.tan(rad_angle) * (0 - orig_x) + orig_y
    x2, y2 = width, np.tan(rad_angle) * (width - orig_x) + orig_y
    return [int(x1), int(y1), int(x2), int(y2)]
