"""
parameter_sweep.py

Investigates how the tunable parameters of two segmentation methods
affect results, as required by Question 4's tasks:

    - "Implement region-growing segmentation and investigate the effect
       of seed-point selection and similarity criteria."
    - "Implement K-means clustering ... and investigate the effect of
       different values of K."

For region growing, sweeps the intensity-similarity threshold used when
growing from each seed (a tighter threshold = stricter similarity
criterion = smaller/fragmented regions; a looser threshold = regions
bleed into background/shadows). Seed *selection* itself already comes
from an automatic distance-transform step in segmentation_pipeline.py;
this script reports how many seeds that step finds per image, since
seed count/placement is what "seed-point selection" drives.

For K-means, sweeps K itself.

For each parameter value, this script re-segments every LABELED image
(the ones with ground-truth masks in masks/) and records IoU, precision,
recall, F1, detected object count, and processing time -- so you can
show both segmentation quality and object-count accuracy as a function
of the parameter.

Run this AFTER label_masks.py (needs masks/coin_XXX_mask.png).

Usage:
    python parameter_sweep.py

Outputs:
    results/sweep_region_growing_threshold.csv
    results/sweep_kmeans_k.csv
    results/plots/sweep_region_growing_threshold.png
    results/plots/sweep_kmeans_k.png
    results/plots/sweep_kmeans_k_qualitative_<image>.png   (visual K comparison)
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from segmentation_pipeline import (
    preprocess,
    segment_region_growing,
    segment_kmeans,
    get_seed_points,
    otsu_best_polarity,
    measure_objects,
    draw_overlay,
)
from evaluate_segmentation import compute_metrics, load_gt_masks

IMG_DIR = "dataset"
OUT_DIR = "results"
PLOT_DIR = os.path.join(OUT_DIR, "plots")

REGION_GROW_THRESHOLDS = [5, 10, 15, 20, 25, 30]
KMEANS_K_VALUES = [2, 3, 4, 5, 6]


# --------------------------------------------------------------------------
# Region growing threshold sweep
# --------------------------------------------------------------------------

def sweep_region_growing(gt_masks):
    rows = []
    for fname, gt_mask in gt_masks.items():
        img_path = os.path.join(IMG_DIR, fname)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        gray, blurred = preprocess(img_bgr)

        # Report seed count once per image (seed *selection* is automatic;
        # this shows how many candidate coin centers it finds).
        rough = otsu_best_polarity(blurred)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        rough = cv2.morphologyEx(rough, cv2.MORPH_OPEN, kernel, iterations=2)
        n_seeds = len(get_seed_points(rough))

        for thresh in REGION_GROW_THRESHOLDS:
            start = time.time()
            mask = segment_region_growing(blurred, thresh=thresh)
            elapsed = time.time() - start

            iou, prec, rec, f1 = compute_metrics(mask, gt_mask)
            objects = measure_objects(mask)
            rows.append({
                "filename": fname, "threshold": thresh, "n_seeds": n_seeds,
                "num_detected": len(objects), "iou": iou,
                "precision": prec, "recall": rec, "f1": f1,
                "processing_time_s": round(elapsed, 4),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# K-means K sweep
# --------------------------------------------------------------------------

def sweep_kmeans(gt_masks):
    rows = []
    for fname, gt_mask in gt_masks.items():
        img_path = os.path.join(IMG_DIR, fname)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        for k in KMEANS_K_VALUES:
            start = time.time()
            mask = segment_kmeans(img_bgr, k=k)
            elapsed = time.time() - start

            iou, prec, rec, f1 = compute_metrics(mask, gt_mask)
            objects = measure_objects(mask)
            rows.append({
                "filename": fname, "k": k,
                "num_detected": len(objects), "iou": iou,
                "precision": prec, "recall": rec, "f1": f1,
                "processing_time_s": round(elapsed, 4),
            })
    return pd.DataFrame(rows)


def save_kmeans_qualitative_examples(gt_masks, k_values=(2, 3, 4, 6), max_images=2):
    """Side-by-side visual comparison of K-means output at different K,
    for a couple of representative images -- useful evidence in the report."""
    for fname in list(gt_masks.keys())[:max_images]:
        img_path = os.path.join(IMG_DIR, fname)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        fig, axes = plt.subplots(1, len(k_values), figsize=(4 * len(k_values), 4))
        for ax, k in zip(axes, k_values):
            mask = segment_kmeans(img_bgr, k=k)
            objects = measure_objects(mask)
            vis = draw_overlay(img_bgr, objects, f"K={k}")
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            ax.imshow(vis_rgb)
            ax.set_title(f"K={k} ({len(objects)} objects)")
            ax.axis("off")
        plt.tight_layout()
        out_path = os.path.join(
            PLOT_DIR, f"sweep_kmeans_k_qualitative_{fname.replace('.jpg', '')}.png"
        )
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_region_growing_sweep(df, out_path):
    means = df.groupby("threshold")[["iou", "precision", "recall", "f1"]].mean()
    fig, ax1 = plt.subplots(figsize=(7, 5))
    for col in ["iou", "precision", "recall", "f1"]:
        ax1.plot(means.index, means[col], marker="o", label=col.upper())
    ax1.set_xlabel("Region-growing similarity threshold")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1)
    ax1.set_title("Region Growing: Effect of Similarity Threshold")
    ax1.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_kmeans_sweep(df, out_path):
    means = df.groupby("k")[["iou", "precision", "recall", "f1"]].mean()
    fig, ax1 = plt.subplots(figsize=(7, 5))
    for col in ["iou", "precision", "recall", "f1"]:
        ax1.plot(means.index, means[col], marker="o", label=col.upper())
    ax1.set_xlabel("K (number of clusters)")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1)
    ax1.set_title("K-means: Effect of K")
    ax1.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    gt_masks = load_gt_masks()
    if not gt_masks:
        print("No ground-truth masks found in masks/. Run label_masks.py first.")
        return
    print(f"Loaded {len(gt_masks)} ground-truth masks. Running sweeps...")

    print("\nSweeping region-growing similarity threshold:", REGION_GROW_THRESHOLDS)
    rg_df = sweep_region_growing(gt_masks)
    rg_df.to_csv(os.path.join(OUT_DIR, "sweep_region_growing_threshold.csv"), index=False)
    print(rg_df.groupby("threshold")[["iou", "num_detected"]].mean())
    plot_region_growing_sweep(rg_df, os.path.join(PLOT_DIR, "sweep_region_growing_threshold.png"))

    print("\nSweeping K-means K:", KMEANS_K_VALUES)
    km_df = sweep_kmeans(gt_masks)
    km_df.to_csv(os.path.join(OUT_DIR, "sweep_kmeans_k.csv"), index=False)
    print(km_df.groupby("k")[["iou", "num_detected"]].mean())
    plot_kmeans_sweep(km_df, os.path.join(PLOT_DIR, "sweep_kmeans_k.png"))

    print("\nSaving qualitative K-means comparison images...")
    save_kmeans_qualitative_examples(gt_masks)

    print(f"\nDone. CSVs in {OUT_DIR}/, plots in {PLOT_DIR}/")


if __name__ == "__main__":
    main()
