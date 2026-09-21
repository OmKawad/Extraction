"""
segment_part.py
----------------
Core classical Computer Vision pipeline for extracting an automotive part
region from a raw mobile-camera photo:

    1. Read image
    2. Grayscale conversion
    3. Gaussian Blur
    4. Canny Edge Detection
    5. Morphological Closing
    6. Largest Contour Detection
    7. Contour Refinement (approxPolyDP + optional convexHull)
    8. Binary Mask generation
    9. Part extraction (segmented + cropped + contour overlay)

No deep learning / no external trained models -- pure OpenCV + NumPy.

The main entry point is `process_image(path)`, which returns a dict holding
every intermediate result (for visualization) plus the final outputs.
"""

import time
import cv2
import numpy as np

import config
import utils


def load_and_preprocess(image_bgr):
    """
    Steps 1-3: resize for processing, add a border, grayscale, Gaussian blur.

    Returns a dict with every intermediate array plus bookkeeping needed to
    map results back to the original resolution later.
    """
    resized, scale = utils.resize_for_processing(image_bgr)

    border = config.ADD_BORDER_PX
    if config.BORDER_MODE == "constant":
        bordered = cv2.copyMakeBorder(
            resized, border, border, border, border,
            cv2.BORDER_CONSTANT, value=config.BORDER_COLOR
        )
    else:
        bordered = cv2.copyMakeBorder(
            resized, border, border, border, border, cv2.BORDER_REPLICATE
        )

    gray = cv2.cvtColor(bordered, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, config.GAUSSIAN_KERNEL_SIZE, config.GAUSSIAN_SIGMA)

    return {
        "resized": resized,
        "scale": scale,
        "border": border,
        "bordered": bordered,
        "gray": gray,
        "blurred": blurred,
    }


def detect_edges(blurred):
    """Step 4: Canny edge detection with optional auto-thresholding."""
    if config.USE_AUTO_CANNY:
        low, high = utils.auto_canny(blurred)
    else:
        low, high = config.CANNY_LOW_THRESHOLD, config.CANNY_HIGH_THRESHOLD

    edges = cv2.Canny(
        blurred, low, high,
        apertureSize=config.CANNY_APERTURE_SIZE,
        L2gradient=config.CANNY_L2_GRADIENT,
    )
    return edges, (low, high)


def apply_morphological_closing(edges):
    """
    Step 5: Morphological Closing to bridge gaps in the edge map and form
    continuous, closed contours. Optionally followed by a light dilation
    pass, which helps a lot on metal parts with specular-highlight gaps.
    """
    kernel = utils.get_morph_kernel(config.MORPH_KERNEL_SHAPE, config.MORPH_KERNEL_SIZE)
    closed = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, kernel, iterations=config.MORPH_CLOSE_ITERATIONS
    )

    if config.APPLY_EXTRA_DILATION:
        dkernel = utils.get_morph_kernel("ellipse", config.DILATION_KERNEL_SIZE)
        closed = cv2.dilate(closed, dkernel, iterations=config.DILATION_ITERATIONS)

    return closed


def find_largest_contour(binary_image, min_area_ratio=None):
    """
    Step 6: Find all contours, filter tiny noise contours, and pick the one
    with maximum area (assumed to be the part).

    Returns (largest_contour_or_None, all_filtered_contours, total_area_px).
    """
    if min_area_ratio is None:
        min_area_ratio = config.MIN_CONTOUR_AREA_RATIO

    retrieval = cv2.RETR_EXTERNAL if config.CONTOUR_RETRIEVAL_MODE == "external" else cv2.RETR_LIST
    contours, _ = cv2.findContours(binary_image, retrieval, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, [], 0

    image_area = binary_image.shape[0] * binary_image.shape[1]
    min_area = image_area * min_area_ratio

    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not filtered:
        return None, [], 0

    largest = max(filtered, key=cv2.contourArea)
    return largest, filtered, image_area


def otsu_fallback_contour(gray, min_area_ratio=None):
    """
    Fallback path used only if Canny + morphology found nothing usable.
    Uses Otsu's threshold to binarize, tries both polarities (part could be
    darker or lighter than the background), and returns the largest contour.
    """
    if min_area_ratio is None:
        min_area_ratio = config.MIN_CONTOUR_AREA_RATIO

    _, th_normal = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_contour, best_area = None, 0
    image_area = gray.shape[0] * gray.shape[1]
    min_area = image_area * min_area_ratio

    for th in (th_normal, th_inv):
        kernel = utils.get_morph_kernel("ellipse", config.MORPH_KERNEL_SIZE)
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_area and area > best_area:
                best_contour, best_area = c, area

    return best_contour


def refine_contour(contour):
    """
    Step 7: Smooth/clean the raw contour.
    - approxPolyDP removes small jagged noise while preserving overall shape
    - convexHull (optional, config-controlled) further smooths outline noise
      from reflections; disabled-by-force for concave parts like brackets.
    """
    perimeter = cv2.arcLength(contour, True)
    epsilon = config.APPROX_POLY_EPSILON_FACTOR * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)

    refined = approx
    used_hull = False
    if config.USE_CONVEX_HULL or config.FORCE_CONVEX_HULL:
        hull = cv2.convexHull(approx)
        # Only prefer the hull if it doesn't inflate area drastically, unless
        # the config forces it -- keeps concave bracket shapes intact.
        hull_area = cv2.contourArea(hull)
        approx_area = cv2.contourArea(approx)
        if config.FORCE_CONVEX_HULL or (approx_area > 0 and hull_area / approx_area < 1.15):
            refined = hull
            used_hull = True

    return refined, used_hull


def build_mask(shape_hw, contour):
    """
    Step 8: Rasterize the refined contour into a filled binary mask
    (255 = part, 0 = background), then smooth the boundary slightly to
    remove pixel-level jaggedness.
    """
    mask = np.zeros(shape_hw, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    if config.MASK_SMOOTH_BLUR_KERNEL and config.MASK_SMOOTH_BLUR_KERNEL != (0, 0):
        mask = cv2.GaussianBlur(mask, config.MASK_SMOOTH_BLUR_KERNEL, 0)
        _, mask = cv2.threshold(mask, config.MASK_SMOOTH_THRESHOLD, 255, cv2.THRESH_BINARY)

    return mask


def extract_segmented(original_bgr, mask, background=None):
    """
    Step 9a: Apply the mask to the full-resolution original image, removing
    the background according to config.SEGMENTED_BACKGROUND.
    """
    if background is None:
        background = config.SEGMENTED_BACKGROUND

    if background == "transparent":
        b, g, r = cv2.split(original_bgr)
        bgra = cv2.merge([b, g, r, mask])
        return bgra

    result = original_bgr.copy()
    bg_color = (255, 255, 255) if background == "white" else (0, 0, 0)
    result[mask == 0] = bg_color
    return result


def extract_cropped(segmented, bbox, padding, image_shape):
    """Step 9b: Tight crop of the segmented output around the part's bbox."""
    h, w = image_shape[:2]
    x, y, bw, bh = bbox
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(w, x + bw + padding)
    y1 = min(h, y + bh + padding)
    return segmented[y0:y1, x0:x1]


def draw_contour_overlay(original_bgr, largest_contour_orig, refined_contour_orig):
    """Step 9c: Draw the raw largest contour and the refined contour on top
    of the original image for visual QA."""
    overlay = original_bgr.copy()
    if largest_contour_orig is not None:
        cv2.drawContours(overlay, [largest_contour_orig], -1,
                          config.VIS_LARGEST_CONTOUR_COLOR, config.VIS_CONTOUR_THICKNESS)
    if refined_contour_orig is not None:
        cv2.drawContours(overlay, [refined_contour_orig], -1,
                          config.VIS_REFINED_CONTOUR_COLOR, config.VIS_CONTOUR_THICKNESS)
    return overlay


def process_image(path):
    """
    Run the full pipeline on a single image file.

    Returns a result dict with keys:
      success, filename, message,
      original, gray, blurred, edges, closed,
      largest_contour_orig, refined_contour_orig, used_hull,
      mask, bbox, cropped, segmented, contour_overlay,
      area_ratio, num_contours_found, canny_thresholds, processing_time_sec
    """
    t0 = time.time()
    filename = utils.basename_no_ext(path)
    result = {"filename": filename, "success": False, "message": ""}

    try:
        original = utils.read_image(path)
    except IOError as e:
        result["message"] = str(e)
        result["processing_time_sec"] = time.time() - t0
        return result

    result["original"] = original
    h_orig, w_orig = original.shape[:2]

    # --- Steps 1-3 ---
    pre = load_and_preprocess(original)
    result["gray"] = pre["gray"]
    result["blurred"] = pre["blurred"]

    # --- Step 4 ---
    edges, canny_thresh = detect_edges(pre["blurred"])
    result["edges"] = edges
    result["canny_thresholds"] = canny_thresh

    # --- Step 5 ---
    closed = apply_morphological_closing(edges)
    result["closed"] = closed

    # --- Step 6 ---
    largest, filtered_contours, proc_area = find_largest_contour(closed)
    used_fallback = False

    if largest is None and config.ENABLE_OTSU_FALLBACK:
        used_fallback = True
        largest = otsu_fallback_contour(pre["gray"])
        filtered_contours = [largest] if largest is not None else []

    result["num_contours_found"] = len(filtered_contours)
    result["used_otsu_fallback"] = used_fallback

    if largest is None:
        result["message"] = (
            "No part contour found. The part may lack contrast against the "
            "background, or MIN_CONTOUR_AREA_RATIO may be too high."
        )
        result["processing_time_sec"] = time.time() - t0
        return result

    # Map the raw largest contour back to original-resolution coordinates
    # (for the visualization overlay).
    largest_orig = utils.map_points_to_original(
        largest, pre["scale"], pre["border"], original.shape
    )
    result["largest_contour_orig"] = largest_orig

    # --- Step 7 ---
    refined, used_hull = refine_contour(largest)
    result["used_hull"] = used_hull
    refined_orig = utils.map_points_to_original(
        refined, pre["scale"], pre["border"], original.shape
    )
    result["refined_contour_orig"] = refined_orig

    # --- Step 8 ---
    mask = build_mask((h_orig, w_orig), refined_orig)
    result["mask"] = mask

    part_area_px = int(cv2.countNonZero(mask))
    result["area_ratio"] = part_area_px / float(h_orig * w_orig)

    # --- Step 9 ---
    bbox = cv2.boundingRect(refined_orig)
    result["bbox"] = bbox

    segmented = extract_segmented(original, mask)
    result["segmented"] = segmented

    cropped = extract_cropped(segmented, bbox, config.CROP_PADDING_PX, original.shape)
    result["cropped"] = cropped

    overlay = draw_contour_overlay(original, largest_orig, refined_orig)
    result["contour_overlay"] = overlay

    result["success"] = True
    result["message"] = "OK" + (" (Otsu fallback used)" if used_fallback else "")
    result["processing_time_sec"] = time.time() - t0
    return result
