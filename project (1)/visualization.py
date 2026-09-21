"""
visualization.py
-----------------
Builds the step-by-step visualization grid:

    Original -> Edges -> Morph Closing -> Largest Contour ->
    Refined Contour -> Binary Mask -> Final Segmented -> Contour Overlay

Pure OpenCV/NumPy image compositing (no matplotlib dependency).
"""

import cv2
import numpy as np

import config


def _to_bgr(image):
    """Normalize any of grayscale / BGR / BGRA into a 3-channel BGR image."""
    if image is None:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        # Composite the transparent (alpha) image over a mid-gray checkerboard
        # so transparency is visible in the flattened visualization panel.
        bgr = image[:, :, :3].astype(np.float32)
        alpha = (image[:, :, 3:4].astype(np.float32)) / 255.0
        bg = _checkerboard(image.shape[0], image.shape[1])
        composited = bgr * alpha + bg.astype(np.float32) * (1 - alpha)
        return composited.astype(np.uint8)

    return image


def _checkerboard(h, w, tile=12):
    board = np.full((h, w, 3), 200, dtype=np.uint8)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if ((x // tile) + (y // tile)) % 2 == 0:
                board[y:y + tile, x:x + tile] = 150
    return board


def _panel(title, image, width=None, body_height=None):
    """
    Resize image to fit within (width, body_height) preserving aspect ratio,
    center it on a white canvas of exactly that size, and stamp a title bar
    above it. A fixed body_height (shared across all panels in a grid) is
    what keeps every panel exactly the same size so they can be concatenated,
    even though inputs (bordered edge maps vs. original-resolution masks)
    have slightly different aspect ratios.
    """
    if width is None:
        width = config.VIS_THUMB_WIDTH

    bgr = _to_bgr(image)
    h, w = bgr.shape[:2]

    if body_height is None:
        body_height = max(1, int(round(width * h / float(w))))

    scale = min(width / float(w), body_height / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    title_bar_h = 30
    panel = np.full((body_height + title_bar_h, width, 3), 255, dtype=np.uint8)

    # center the resized image within the body area
    y_off = title_bar_h + (body_height - new_h) // 2
    x_off = (width - new_w) // 2
    panel[y_off:y_off + new_h, x_off:x_off + new_w, :] = resized

    cv2.putText(
        panel, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
        config.VIS_FONT_SCALE, (20, 20, 20), config.VIS_FONT_THICKNESS, cv2.LINE_AA
    )
    cv2.rectangle(panel, (0, 0), (width - 1, panel.shape[0] - 1), (180, 180, 180), 1)
    return panel


def _stack_grid(panels, cols):
    """Arrange a list of equally-sized panels into a grid with `cols` columns."""
    rows = []
    for i in range(0, len(panels), cols):
        row_panels = panels[i:i + cols]
        # pad the last row if it's short
        while len(row_panels) < cols:
            blank = np.full_like(row_panels[0], 255)
            row_panels.append(blank)
        rows.append(cv2.hconcat(row_panels))
    return cv2.vconcat(rows)


def build_pipeline_grid(result, cols=4):
    """
    Build the full 8-panel step-by-step visualization for one processed image.
    `result` is the dict returned by segment_part.process_image().
    """
    original = result.get("original")
    edges = result.get("edges")
    closed = result.get("closed")
    mask = result.get("mask")
    segmented = result.get("segmented")

    largest_vis = original.copy() if original is not None else None
    if largest_vis is not None and result.get("largest_contour_orig") is not None:
        cv2.drawContours(largest_vis, [result["largest_contour_orig"]], -1,
                          config.VIS_LARGEST_CONTOUR_COLOR, config.VIS_CONTOUR_THICKNESS)

    refined_vis = original.copy() if original is not None else None
    if refined_vis is not None and result.get("refined_contour_orig") is not None:
        cv2.drawContours(refined_vis, [result["refined_contour_orig"]], -1,
                          config.VIS_REFINED_CONTOUR_COLOR, config.VIS_CONTOUR_THICKNESS)

    width = config.VIS_THUMB_WIDTH
    oh, ow = original.shape[:2]
    body_height = max(1, int(round(width * oh / float(ow))))

    panels = [
        _panel("1. Original", original, width, body_height),
        _panel("2. Canny Edges", edges, width, body_height),
        _panel("3. Morph Closing", closed, width, body_height),
        _panel("4. Largest Contour", largest_vis, width, body_height),
        _panel("5. Refined Contour", refined_vis, width, body_height),
        _panel("6. Binary Mask", mask, width, body_height),
        _panel("7. Final Segmented", segmented, width, body_height),
        _panel("8. Contour Overlay", result.get("contour_overlay"), width, body_height),
    ]

    return _stack_grid(panels, cols)
