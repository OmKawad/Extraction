# Automotive Part Segmentation — Classical Computer Vision Pipeline

A pure OpenCV + NumPy pipeline that takes a raw mobile-camera photo of an
automotive supplier part (sheet metal, bracket, machined component, etc.)
and outputs the part region with the background removed. **No deep learning,
no trained models, no GPU required.**

---

## 1. How it works

```
Read Image
    │
    ▼
Grayscale Conversion
    │
    ▼
Gaussian Blur                  (reduce sensor/JPEG noise before edge detection)
    │
    ▼
Canny Edge Detection           (gradient-magnitude auto-thresholded, see §4)
    │
    ▼
Morphological Closing          (bridge gaps, connect broken edges into loops)
    │
    ▼
Largest External Contour       (filtered by min-area to reject noise specks)
    │
    ▼
Contour Refinement             (approxPolyDP smoothing + optional convexHull)
    │
    ▼
Binary Mask (white=part, black=background)
    │
    ▼
Part Extraction → Segmented / Cropped / Contour-overlay outputs
```

Every image is downscaled for processing (default: longest side ≤ 1600 px)
for speed and more stable edge detection, but **all masks, contours and
outputs are mapped back to full original resolution** before saving —
nothing about the final output is downgraded.

---

## 2. Project structure

```
project/
├── config.py                 # every tunable parameter, documented inline
├── utils.py                  # I/O, resize/rescale helpers, auto-Canny, kernels
├── segment_part.py           # the 9-step pipeline (single image)
├── visualization.py          # builds the step-by-step visual grid
├── batch_process.py          # entry point — run this
├── requirements.txt
├── README.md
├── dataset_raw/              # <- put your input images here
└── outputs/                  # created automatically on first run
    ├── masks/                #   <name>_mask.png            (binary, white=part)
    ├── cropped_parts/        #   <name>_cropped.png          (tight crop, bg removed)
    ├── segmented_parts/      #   <name>_segmented.png        (full-size, bg removed)
    ├── contour_visualizations/
    │       <name>_contour_overlay.png    (original + drawn contours)
    │       <name>_pipeline_steps.png     (8-panel step-by-step grid)
    └── reports/
            summary_report.csv
            summary_report.txt
```

---

## 3. Setup & usage

```bash
pip install -r requirements.txt

# put your images in dataset_raw/ -- .jpg, .png, .bmp, .tif, .heic, .heif all work
cp /path/to/your/photos/*.heic dataset_raw/

python batch_process.py
```

### A note on HEIC/HEIF (iPhone photos)

iPhones save photos as `.heic` by default. OpenCV can't decode HEIC directly,
so this project reads it through Pillow + `pillow-heif` instead (see
`utils.read_image()`) and converts it to a normal BGR array before handing
it to the rest of the pipeline — everything downstream (Canny, contours,
masks, outputs) works exactly the same regardless of the input format.

EXIF orientation is also corrected automatically: phones frequently store
the pixels sideways/upside-down and rely on an EXIF tag to display them
right-side-up. This project applies that correction on load
(`PIL.ImageOps.exif_transpose`), so the image is processed and saved
right-side-up, the same way it looks in your phone's gallery.

If you'd rather not add the extra dependency, convert HEIC to JPG/PNG first
(e.g. with `sips` on macOS, or any HEIC converter) and drop those into
`dataset_raw/` instead — no code changes needed either way.

Console output looks like:

```
Found 12 image(s) in 'dataset_raw'.
======================================================================
[1/12] Processing: bracket_0001.jpg
    [OK] area_ratio=0.2841  bbox=(412, 205, 1580, 1122)  time=0.31s
...
Done. 11 succeeded, 1 failed. Total time: 4.02s
Reports written to: outputs/reports/summary_report.csv
                    outputs/reports/summary_report.txt
```

`summary_report.csv` logs, per image: status, number of contours found,
whether the Otsu fallback or convex hull was used, area ratio, bounding
box, the Canny thresholds actually used, and processing time — useful for
auditing a batch run or spotting which images need re-shooting.

---

## 4. Why gradient-magnitude auto-Canny (important design choice)

A very common "auto-Canny" recipe derives thresholds from the **median
pixel intensity** of the image. That heuristic is tuned for textured
natural photos and **fails badly on industrial part images**, which are
mostly two large flat regions (uniform part + uniform background) with a
comparatively low-contrast boundary between them. On that kind of image,
median intensity has almost no relationship to the actual edge gradient
strength, and the derived thresholds end up either far too high (no edges
detected at all → the whole frame gets picked up by the Otsu fallback) or
too low (background texture/noise floods the edge map).

This pipeline instead computes the image's own **Sobel gradient-magnitude
distribution** and sets:

```
upper_threshold = percentile(gradient_magnitude, AUTO_CANNY_GRADIENT_PERCENTILE)
lower_threshold = upper_threshold * AUTO_CANNY_LOWER_RATIO
```

This tracks the actual boundary strength in each photo and is robust to
exposure/lighting changes between shots. See `utils.auto_canny()`.

---

## 5. Tuning guide for automotive parts

All parameters live in `config.py` with inline explanations. The most
impactful ones for this domain:

| Symptom | Likely cause | Fix |
|---|---|---|
| Whole image (or huge rectangle) selected as the part | Background too similar to a fixed border color, or Otsu fallback triggered on a near-uniform scene | Keep `BORDER_MODE = "replicate"` (default); increase `AUTO_CANNY_GRADIENT_PERCENTILE` if noise floods edges |
| Part boundary has small chips/nothing detected | Edge too faint (low contrast, motion blur) | Lower `AUTO_CANNY_GRADIENT_PERCENTILE` (e.g. 85–90) or `AUTO_CANNY_MIN_UPPER`; increase `MORPH_KERNEL_SIZE` to bridge bigger gaps |
| Small bolt holes / dents get filled in wrongly | `MORPH_CLOSE_ITERATIONS` too high, gaps in edge map | Reduce `MORPH_KERNEL_SIZE` / iterations |
| Mask edges look jagged / staircase-y | Insufficient smoothing | Increase `MASK_SMOOTH_BLUR_KERNEL` |
| L-shaped / concave brackets get "filled in" square | Convex hull forced on a concave part | Keep `FORCE_CONVEX_HULL = False` (default); `USE_CONVEX_HULL = True` already only applies the hull when it doesn't inflate area much |
| Specular highlights (glossy metal) break the outline | Local edge dropout at bright reflections | `APPLY_EXTRA_DILATION = True` (default) with a slightly larger `DILATION_KERNEL_SIZE` |
| Small background clutter picked up as "largest contour" | `MIN_CONTOUR_AREA_RATIO` too low | Raise it (e.g. 0.02–0.05) |
| Part touching / cropped by the frame edge | Border merges part boundary with image edge | `ADD_BORDER_PX` + `BORDER_MODE="replicate"` already mitigates this; increase `ADD_BORDER_PX` if still an issue |

**Recommended capture setup** for best classical-CV results: matte,
contrasting background (e.g. dark non-reflective mat under light metal
parts, or vice versa), diffuse/non-directional lighting to minimize
specular streaks, and the part fully inside the frame with margin on all
sides.

---

## 6. Output types explained

- **Binary Mask** (`masks/`): single-channel, white = part, black =
  background, full original resolution.
- **Segmented Part** (`segmented_parts/`): full-canvas image, same size as
  the original, background removed (transparent RGBA by default —
  configurable to solid black/white via `SEGMENTED_BACKGROUND`).
- **Cropped Part** (`cropped_parts/`): the segmented output tightly cropped
  to the part's bounding box (+ `CROP_PADDING_PX` margin).
- **Contour Overlay** (`contour_visualizations/*_contour_overlay.png`): the
  original image with the raw largest contour (orange) and refined contour
  (green) drawn on top — for visual QA of detection accuracy.
- **Pipeline Steps** (`contour_visualizations/*_pipeline_steps.png`): an
  8-panel grid showing every stage of the pipeline for that image, useful
  for debugging a misdetection.

---

## 7. Known limitations (inherent to classical CV, not a bug)

- Assumes **one dominant part per image** — the pipeline extracts the single
  largest contour. Multiple parts in one frame will only yield the biggest.
- Needs **some intensity/gradient contrast** between part and background.
  A part that matches the background almost exactly in color, texture, and
  lighting will not segment reliably by edges/contours alone — reshoot on a
  contrasting mat is the practical fix, not a parameter tweak.
- Very cluttered or textured backgrounds (e.g. a busy workbench) can
  confuse the largest-contour assumption if the background contour itself
  exceeds `MIN_CONTOUR_AREA_RATIO`; use a plain backdrop for production runs.
- Transparent/glass or highly reflective/mirror-finish parts are the
  hardest case for any edge-based method — reflections create false edges.

If accuracy on a particularly difficult part family plateaus even after
tuning `config.py`, that's usually the point where a learned segmentation
model (which this project intentionally avoids) starts to outperform
classical edge/contour methods — worth flagging back to stakeholders as a
data point, not something to fight further in this pipeline.

---

## 8. Extending

- **Multiple parts per image**: change `find_largest_contour` in
  `segment_part.py` to return all contours above the area threshold instead
  of only the max, and loop `extract_segmented`/`extract_cropped` per contour.
- **Different file naming**: adjust the `f"{filename}_..."` patterns in
  `batch_process.py`.
- **JSON report instead of / in addition to CSV**: `batch_process._write_reports`
  is a small, isolated function — easy to extend.
