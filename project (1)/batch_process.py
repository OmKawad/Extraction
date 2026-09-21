"""
batch_process.py
-----------------
Entry point. Processes every image in dataset_raw/ through the classical CV
segmentation pipeline and writes results to outputs/.

Usage:
    python batch_process.py
"""

import os
import csv
import time

import config
import utils
import segment_part
import visualization


def process_dataset():
    utils.ensure_dirs(config.ALL_OUTPUT_DIRS)

    image_paths = utils.list_images(config.INPUT_DIR)
    if not image_paths:
        utils.log(f"[!] No images found in '{config.INPUT_DIR}'.")
        utils.log(f"    Supported extensions: {config.SUPPORTED_EXTENSIONS}")
        utils.log(f"    Create the folder and add images, then re-run.")
        os.makedirs(config.INPUT_DIR, exist_ok=True)
        return

    utils.log(f"Found {len(image_paths)} image(s) in '{config.INPUT_DIR}'.")
    utils.log("=" * 70)

    report_rows = []
    n_success, n_failed = 0, 0
    t_start = time.time()

    for idx, path in enumerate(image_paths, start=1):
        filename = utils.basename_no_ext(path)
        utils.log(f"[{idx}/{len(image_paths)}] Processing: {os.path.basename(path)}")

        result = segment_part.process_image(path)

        row = {
            "filename": os.path.basename(path),
            "status": "SUCCESS" if result["success"] else "FAILED",
            "message": result.get("message", ""),
            "num_contours_found": result.get("num_contours_found", 0),
            "used_otsu_fallback": result.get("used_otsu_fallback", False),
            "used_convex_hull": result.get("used_hull", False),
            "area_ratio": round(result.get("area_ratio", 0.0), 4),
            "bbox_xywh": result.get("bbox", ""),
            "canny_thresholds": result.get("canny_thresholds", ""),
            "processing_time_sec": round(result.get("processing_time_sec", 0.0), 3),
        }
        report_rows.append(row)

        if not result["success"]:
            n_failed += 1
            utils.log(f"    [FAILED] {result.get('message')}")
            continue

        n_success += 1

        # --- Save all outputs ---
        mask_path = os.path.join(config.MASK_DIR, f"{filename}_mask.png")
        utils.save_image(mask_path, result["mask"])

        cropped_path = os.path.join(config.CROPPED_DIR, f"{filename}_cropped.png")
        utils.save_image(cropped_path, result["cropped"])

        segmented_path = os.path.join(config.SEGMENTED_DIR, f"{filename}_segmented.png")
        utils.save_image(segmented_path, result["segmented"])

        overlay_path = os.path.join(config.CONTOUR_VIS_DIR, f"{filename}_contour_overlay.png")
        utils.save_image(overlay_path, result["contour_overlay"])

        grid = visualization.build_pipeline_grid(result)
        grid_path = os.path.join(config.CONTOUR_VIS_DIR, f"{filename}_pipeline_steps.png")
        utils.save_image(grid_path, grid)

        utils.log(
            f"    [OK] area_ratio={row['area_ratio']}  "
            f"bbox={row['bbox_xywh']}  time={row['processing_time_sec']}s"
        )

    total_time = time.time() - t_start
    utils.log("=" * 70)
    utils.log(f"Done. {n_success} succeeded, {n_failed} failed. "
              f"Total time: {total_time:.2f}s")

    _write_reports(report_rows, n_success, n_failed, total_time)


def _write_reports(report_rows, n_success, n_failed, total_time):
    csv_path = os.path.join(config.REPORTS_DIR, "summary_report.csv")
    fieldnames = [
        "filename", "status", "message", "num_contours_found",
        "used_otsu_fallback", "used_convex_hull", "area_ratio",
        "bbox_xywh", "canny_thresholds", "processing_time_sec",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    txt_path = os.path.join(config.REPORTS_DIR, "summary_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Automotive Part Segmentation - Batch Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total images   : {len(report_rows)}\n")
        f.write(f"Succeeded      : {n_success}\n")
        f.write(f"Failed         : {n_failed}\n")
        f.write(f"Total time     : {total_time:.2f}s\n\n")
        for row in report_rows:
            f.write(f"- {row['filename']}: {row['status']}")
            if row["status"] == "FAILED":
                f.write(f" -> {row['message']}")
            else:
                f.write(f" (area_ratio={row['area_ratio']}, bbox={row['bbox_xywh']})")
            f.write("\n")

    utils.log(f"Reports written to: {csv_path}")
    utils.log(f"                    {txt_path}")


if __name__ == "__main__":
    process_dataset()
