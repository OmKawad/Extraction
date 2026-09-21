"""
utils.py
--------
Generic helper functions used across the pipeline:
- File I/O that is safe with unicode paths
- Downscale-for-processing / rescale-back-to-original helpers
- Auto-Canny threshold computation
- Morphological kernel builder
"""

import os
import cv2
import numpy as np

import config

# HEIC/HEIF (iPhone's default camera format since iOS 11) is not decodable
# by OpenCV's imdecode. pillow-heif registers a libheif-backed codec with
# Pillow so we can open it there and hand the pixels to OpenCV as a normal
# BGR array. This registration is optional at import time so the rest of
# the pipeline still works (for jpg/png/etc.) even if pillow-heif isn't
# installed -- read_image() raises a clear, actionable error only if a
# .heic/.heif file is actually encountered without it.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIF_AVAILABLE = True
except ImportError:
    _HEIF_AVAILABLE = False

from PIL import Image, ImageOps

_HEIF_EXTENSIONS = (".heic", ".heif")


def ensure_dirs(dir_list):
    """Create every directory in dir_list if it doesn't already exist."""
    for d in dir_list:
        os.makedirs(d, exist_ok=True)


def list_images(input_dir):
    """Return sorted list of full paths to supported images in input_dir."""
    if not os.path.isdir(input_dir):
        return []
    files = []
    for f in sorted(os.listdir(input_dir)):
        if f.lower().endswith(config.SUPPORTED_EXTENSIONS):
            files.append(os.path.join(input_dir, f))
    return files


def read_image(path):
    """
    Read an image as BGR (uint8, HxWx3), regardless of source format.

    HEIC/HEIF files are decoded through Pillow + pillow-heif (OpenCV cannot
    read them directly) and EXIF orientation is applied so a photo that a
    phone stored sideways/upside-down comes out right-side-up -- the same
    way it appears in the phone's own gallery. Every other format goes
    through cv2.imdecode(fromfile(...)) instead of cv2.imread, which is more
    reliable with unicode/non-ASCII paths.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in _HEIF_EXTENSIONS:
        if not _HEIF_AVAILABLE:
            raise IOError(
                f"Cannot read '{path}': HEIC/HEIF support needs the "
                f"'pillow-heif' package. Install it with: "
                f"pip install pillow-heif"
            )
        pil_image = Image.open(path)
        pil_image = ImageOps.exif_transpose(pil_image)  # correct phone rotation
        pil_image = pil_image.convert("RGB")
        rgb = np.array(pil_image)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Could not read image (corrupt or unsupported): {path}")
    return image


def save_image(path, image):
    """Encode and write an image (handles BGR and BGRA) to disk."""
    ext = os.path.splitext(path)[1]
    if ext == "":
        ext = ".png"
        path = path + ext
    success, buf = cv2.imencode(ext, image)
    if not success:
        raise IOError(f"Could not encode image for saving: {path}")
    buf.tofile(path)


def resize_for_processing(image, max_dim=None):
    """
    Downscale image so its longest side == max_dim (keeps aspect ratio).
    Returns (resized_image, scale_factor). scale_factor == 1.0 if no resize
    was needed. Multiply processing-resolution coordinates by (1/scale_factor)
    -- i.e. divide by scale_factor -- to map back to original resolution.
    """
    if max_dim is None:
        max_dim = config.RESIZE_MAX_DIM
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image.copy(), 1.0
    scale = max_dim / float(longest)
    resized = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))),
                          interpolation=cv2.INTER_AREA)
    return resized, scale


def map_points_to_original(points, scale, border_px, original_shape):
    """
    Map contour points found on a (bordered + resized) working image back to
    original image coordinates: subtract the border offset, then divide by
    the resize scale. Clips to the original image bounds.
    """
    h, w = original_shape[:2]
    pts = points.astype(np.float32).reshape(-1, 2)
    pts -= border_px
    pts /= scale
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts.astype(np.int32).reshape(-1, 1, 2)


def get_morph_kernel(shape=None, size=None):
    """Build a structuring element from config-friendly string names."""
    if shape is None:
        shape = config.MORPH_KERNEL_SHAPE
    if size is None:
        size = config.MORPH_KERNEL_SIZE
    shape_map = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    return cv2.getStructuringElement(shape_map.get(shape, cv2.MORPH_ELLIPSE), size)


def auto_canny(gray, percentile=None, floor=None, lower_ratio=None):
    """
    Compute (lower, upper) Canny thresholds from the image's own Sobel
    gradient-magnitude distribution.

    NOTE: This intentionally does NOT use the common "median pixel
    intensity" auto-Canny heuristic. That heuristic assumes textured natural
    photos and badly over-estimates thresholds on industrial part images,
    which are mostly large flat regions (uniform part + uniform background)
    with a comparatively low-magnitude gradient jump only at the boundary.
    Deriving thresholds from the gradient magnitude itself is robust to that
    flat-region composition and to overall image brightness/exposure.
    """
    if percentile is None:
        percentile = config.AUTO_CANNY_GRADIENT_PERCENTILE
    if floor is None:
        floor = config.AUTO_CANNY_MIN_UPPER
    if lower_ratio is None:
        lower_ratio = config.AUTO_CANNY_LOWER_RATIO

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)

    upper = max(float(np.percentile(magnitude, percentile)), float(floor))
    lower = upper * lower_ratio
    return int(lower), int(upper)


def basename_no_ext(path):
    return os.path.splitext(os.path.basename(path))[0]


def log(msg):
    if config.VERBOSE:
        print(msg)
