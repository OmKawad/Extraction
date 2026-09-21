"""
config.py
---------
Central configuration for the classical Computer Vision part-segmentation
pipeline. Tune these values based on your camera setup, lighting conditions,
and the type of part (sheet metal, bracket, molded component, etc).

No magic numbers live inside the pipeline code -- everything tunable is here.
"""

import os

# ==============================================================================
# PATHS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "dataset_raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
CROPPED_DIR = os.path.join(OUTPUT_DIR, "cropped_parts")
SEGMENTED_DIR = os.path.join(OUTPUT_DIR, "segmented_parts")
CONTOUR_VIS_DIR = os.path.join(OUTPUT_DIR, "contour_visualizations")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

ALL_OUTPUT_DIRS = [MASK_DIR, CROPPED_DIR, SEGMENTED_DIR, CONTOUR_VIS_DIR, REPORTS_DIR]

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".heic", ".heif")
# .heic / .heif (iPhone default camera format) are decoded via pillow-heif,
# not OpenCV -- see utils.read_image(). Requires pillow-heif + Pillow
# (already in requirements.txt).

# ==============================================================================
# PRE-PROCESSING
# ==============================================================================
# Mobile cameras shoot 12MP+ images. We downscale for speed and more stable
# edge detection, then map every contour/mask back to full resolution before
# any file is saved -- so outputs are always full quality.
RESIZE_MAX_DIM = 1600                 # max(width, height) during processing, px

GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 0

# A border is added around the working image before edge detection. This
# stops the part's boundary from merging with the frame edge when the part
# nearly touches it (common in tight mobile close-ups). BORDER_MODE
# "replicate" extends the existing edge pixels outward -- it does NOT
# introduce a new artificial intensity step, so it can't create a spurious
# full-frame contour the way a fixed border color could on a light
# background. Use "constant" only if your background is reliably uniform
# and BORDER_COLOR matches it closely.
ADD_BORDER_PX = 15
BORDER_MODE = "replicate"             # "replicate" | "constant"
BORDER_COLOR = (255, 255, 255)        # only used when BORDER_MODE = "constant"

# ==============================================================================
# EDGE DETECTION (CANNY)
# ==============================================================================
# Auto-Canny computes thresholds per-image from the image's own Sobel
# gradient-magnitude distribution (NOT the median pixel intensity -- see the
# docstring in utils.auto_canny for why that common heuristic fails on
# mostly-flat industrial part images). This makes thresholds robust to
# shot-to-shot lighting/exposure variation on the factory floor.
USE_AUTO_CANNY = True
AUTO_CANNY_GRADIENT_PERCENTILE = 92   # upper threshold = this percentile of |grad|
AUTO_CANNY_MIN_UPPER = 15             # floor, avoids near-zero thresholds on flat images
AUTO_CANNY_LOWER_RATIO = 0.4          # lower threshold = ratio * upper

# Used only when USE_AUTO_CANNY = False (fixed thresholds for a known, controlled setup).
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

CANNY_APERTURE_SIZE = 3
CANNY_L2_GRADIENT = True

# ==============================================================================
# MORPHOLOGICAL CLOSING
# ==============================================================================
MORPH_KERNEL_SHAPE = "ellipse"        # "rect" | "ellipse" | "cross"
MORPH_KERNEL_SIZE = (9, 9)
MORPH_CLOSE_ITERATIONS = 2

# Metal parts often have specular highlights that locally break the Canny
# edge. One light dilation pass after closing helps bridge those gaps.
APPLY_EXTRA_DILATION = True
DILATION_KERNEL_SIZE = (5, 5)
DILATION_ITERATIONS = 1

# ==============================================================================
# CONTOUR DETECTION
# ==============================================================================
MIN_CONTOUR_AREA_RATIO = 0.01         # ignore contours < 1% of image area (noise)
CONTOUR_RETRIEVAL_MODE = "external"   # only the outer boundary of the part matters

# ==============================================================================
# CONTOUR REFINEMENT
# ==============================================================================
APPROX_POLY_EPSILON_FACTOR = 0.004    # fraction of contour perimeter
USE_CONVEX_HULL = True                # hull smooths noisy edges from reflections
# Brackets/L-shaped parts are concave -- forcing a hull would erase the concavity.
# Keep this False unless your parts are always convex (flat blanks, discs, etc).
FORCE_CONVEX_HULL = False

# Smooths the filled mask's boundary (removes staircase/jagged pixel edges).
MASK_SMOOTH_BLUR_KERNEL = (5, 5)
MASK_SMOOTH_THRESHOLD = 127

# ==============================================================================
# EXTRACTION
# ==============================================================================
CROP_PADDING_PX = 10                  # padding around bounding box for crop output
SEGMENTED_BACKGROUND = "transparent"  # "transparent" | "black" | "white"

# ==============================================================================
# FALLBACK (robustness)
# ==============================================================================
# If Canny + morphology finds no valid contour (e.g. very low contrast between
# a light metal part and a light background), fall back to Otsu thresholding
# on the grayscale image before declaring failure.
ENABLE_OTSU_FALLBACK = True

# ==============================================================================
# VISUALIZATION
# ==============================================================================
VIS_THUMB_WIDTH = 380                 # width of each panel in the step-by-step grid
VIS_FONT_SCALE = 0.55
VIS_FONT_THICKNESS = 1
VIS_LARGEST_CONTOUR_COLOR = (0, 165, 255)   # orange, BGR
VIS_REFINED_CONTOUR_COLOR = (0, 255, 0)     # green, BGR
VIS_CONTOUR_THICKNESS = 3

# ==============================================================================
# LOGGING
# ==============================================================================
VERBOSE = True
