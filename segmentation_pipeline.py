"""
segmentation_pipeline.py

Implements three classical segmentation approaches for the coin
counting/measurement task (CSCD608 - Question 4):

    1. Thresholding       (Otsu global threshold)
    2. Region growing     (seeded, intensity-similarity based)
    3. K-means clustering (Lab colour space, K configurable)

For each image + method, this script:
    - preprocesses the image (grayscale, denoise)
    - produces a binary foreground mask
    - extracts per-object measurements via contours
      (count, area, perimeter, centroid, bounding box, enclosing circle)
    - saves a visualisation (contours + bboxes + numbered coins)
    - saves the binary mask (needed later for IoU against your ground truth)
    - records processing time per image/method

Usage:
    pip install opencv-python numpy pandas
    python segmentation_pipeline.py

Run this from the folder that contains 'dataset/' (coin_001.jpg ... coin_023.jpg).
Outputs:
    results/masks/<method>/coin_XXX_mask.png
    results/viz/<method>/coin_XXX_viz.png
    results/segmentation_results.csv   <- one row per (image, method)
"""

import os
import time
import glob
import numpy as np
import pandas as pd
import cv2

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

IMG_DIR = "dataset"
OUT_DIR = "results"
MIN_OBJECT_AREA = 300        # filter out tiny noise contours (pixels^2)
KMEANS_K = 3                 # default K; try 2, 3, 4 for your K-sweep experiment
REGION_GROW_THRESH = 12      # intensity similarity threshold for region growing

METHODS = ["thresholding", "region_growing", "kmeans"]


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def preprocess(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blurred


# --------------------------------------------------------------------------
# Method 1: Thresholding (Otsu + morphology)
# --------------------------------------------------------------------------

def otsu_best_polarity(blurred_gray, target_fraction=0.25):
    """
    Coins can be darker or lighter than the background depending on the
    photo (tile vs. brown wood vs. light wood), so compute both Otsu
    polarities and keep whichever gives a more plausible "coins are a
    minority of the frame" foreground. Shared by thresholding and the
    region-growing seed step so both agree on which side is "coin".
    """
    _, mask_inv = cv2.threshold(
        blurred_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    _, mask_norm = cv2.threshold(
        blurred_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    def fg_fraction(m):
        return np.count_nonzero(m) / m.size

    candidates = [mask_inv, mask_norm]
    return min(candidates, key=lambda m: abs(fg_fraction(m) - target_fraction))


def segment_thresholding(blurred_gray):
    mask = otsu_best_polarity(blurred_gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


# --------------------------------------------------------------------------
# Method 2: Region growing (auto-seeded)
# --------------------------------------------------------------------------

def get_seed_points(mask_hint):
    """
    Find one seed point per likely coin using the distance transform of a
    rough mask (peaks of the distance transform sit near coin centers).
    This avoids requiring manual seed clicks for all 23 images.
    """
    dist = cv2.distanceTransform(mask_hint, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return []
    _, peaks = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    peaks = peaks.astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(peaks)
    seeds = []
    for lbl in range(1, n_labels):
        ys, xs = np.where(labels == lbl)
        if len(xs) == 0:
            continue
        seeds.append((int(np.mean(ys)), int(np.mean(xs))))
    return seeds


def region_growing(gray, seeds, thresh=REGION_GROW_THRESH):
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    visited = np.zeros((h, w), dtype=bool)

    for (sy, sx) in seeds:
        if visited[sy, sx]:
            continue
        seed_val = int(gray[sy, sx])
        stack = [(sy, sx)]
        visited[sy, sx] = True
        while stack:
            y, x = stack.pop()
            mask[y, x] = 255
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    if abs(int(gray[ny, nx]) - seed_val) <= thresh:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    return mask


def segment_region_growing(blurred_gray, thresh=REGION_GROW_THRESH):
    # Bootstrap seed locations from a quick Otsu pass (used only to find
    # plausible coin centers -- the actual mask comes from growing).
    # Uses the same polarity-selection as thresholding, since coins are
    # darker than the background in some photos and lighter in others.
    rough = otsu_best_polarity(blurred_gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    rough = cv2.morphologyEx(rough, cv2.MORPH_OPEN, kernel, iterations=2)

    seeds = get_seed_points(rough)
    if not seeds:
        return np.zeros_like(blurred_gray)

    mask = region_growing(blurred_gray, seeds, thresh=thresh)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


# --------------------------------------------------------------------------
# Method 3: K-means clustering
# --------------------------------------------------------------------------

def segment_kmeans(img_bgr, k=KMEANS_K):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    samples = lab.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, _ = cv2.kmeans(
        samples, k, None, criteria, attempts=5, flags=cv2.KMEANS_PP_CENTERS
    )
    labels = labels.reshape((h, w))

    # Heuristic: the background is usually the largest cluster by pixel
    # count. Treat everything else as foreground (coins + shadow edges).
    counts = np.bincount(labels.flatten(), minlength=k)
    bg_cluster = int(np.argmax(counts))
    remaining = [c for c in range(k) if c != bg_cluster]
    coin_cluster = min(remaining, key=lambda c: counts[c])
    mask = np.where(labels == coin_cluster, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


# --------------------------------------------------------------------------
# Object measurement (shared across all three methods)
# --------------------------------------------------------------------------

def measure_objects(mask, min_area=MIN_OBJECT_AREA):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    objects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        x, y, w, h = cv2.boundingRect(cnt)
        (ecx, ecy), radius = cv2.minEnclosingCircle(cnt)
        objects.append({
            "area": area,
            "perimeter": perimeter,
            "centroid": (cx, cy),
            "bbox": (x, y, w, h),
            "enclosing_circle": ((ecx, ecy), radius),
            "contour": cnt,
        })
    return objects


def draw_overlay(img_bgr, objects, method_name):
    vis = img_bgr.copy()
    for i, obj in enumerate(objects, start=1):
        cv2.drawContours(vis, [obj["contour"]], -1, (0, 255, 0), 2)
        x, y, w, h = obj["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cx, cy = obj["centroid"]
        cv2.circle(vis, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        cv2.putText(
            vis, str(i), (int(cx) - 8, int(cy) - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )
    cv2.putText(
        vis, f"{method_name}: {len(objects)} objects", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 255), 2
    )
    return vis


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def run_method(method, img_bgr, gray, blurred, region_grow_thresh=REGION_GROW_THRESH, k=KMEANS_K):
    start = time.time()
    if method == "thresholding":
        mask = segment_thresholding(blurred)
    elif method == "region_growing":
        mask = segment_region_growing(blurred, thresh=region_grow_thresh)
    elif method == "kmeans":
        mask = segment_kmeans(img_bgr, k=k)
    else:
        raise ValueError(method)
    elapsed = time.time() - start
    return mask, elapsed


def main():
    for method in METHODS:
        os.makedirs(os.path.join(OUT_DIR, "masks", method), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, "viz", method), exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(IMG_DIR, "coin_*.jpg")))
    if not image_paths:
        print(f"No images found in {IMG_DIR}/. Check IMG_DIR path.")
        return

    rows = []
    for path in image_paths:
        fname = os.path.basename(path)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(f"Could not read {path}, skipping.")
            continue
        gray, blurred = preprocess(img_bgr)

        print(f"Processing {fname} ...")
        for method in METHODS:
            mask, elapsed = run_method(method, img_bgr, gray, blurred)
            objects = measure_objects(mask)
            vis = draw_overlay(img_bgr, objects, method)

            mask_path = os.path.join(
                OUT_DIR, "masks", method, fname.replace(".jpg", "_mask.png")
            )
            viz_path = os.path.join(
                OUT_DIR, "viz", method, fname.replace(".jpg", "_viz.png")
            )
            cv2.imwrite(mask_path, mask)
            cv2.imwrite(viz_path, vis)

            total_area = sum(o["area"] for o in objects)
            total_perimeter = sum(o["perimeter"] for o in objects)
            rows.append({
                "filename": fname,
                "method": method,
                "num_detected": len(objects),
                "total_area_px": total_area,
                "total_perimeter_px": total_perimeter,
                "processing_time_s": round(elapsed, 4),
                "mask_path": mask_path,
                "viz_path": viz_path,
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "segmentation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDone. Results written to {csv_path}")
    print(f"Masks in {OUT_DIR}/masks/<method>/, visualisations in {OUT_DIR}/viz/<method>/")


if __name__ == "__main__":
    main()
