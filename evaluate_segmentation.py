"""
evaluate_segmentation.py

Evaluates the three segmentation methods (thresholding, region growing,
K-means) from segmentation_pipeline.py against your hand-labeled ground
truth masks, and produces everything Question 4's "Experimental
Requirement" asks for:

    - Per-image / per-method IoU, Precision, Recall, F1
    - Results broken down by condition (lighting, background, touching)
      using labels.csv
    - The required method-comparison table:
        Segmentation Accuracy | Processing Time | Sensitivity to Noise |
        Sensitivity to Illumination
    - Bar-chart visualisations of all of the above

Run this AFTER segmentation_pipeline.py (needs results/masks/<method>/
and results/segmentation_results.csv) and AFTER label_masks.py (needs
masks/coin_XXX_mask.png ground truth for the labeled subset).

Usage:
    pip install opencv-python numpy pandas matplotlib
    python evaluate_segmentation.py

Run from the same folder as segmentation_pipeline.py (this script
imports its segmentation functions directly so the sensitivity test
uses the exact same algorithms).

Outputs:
    results/eval_per_image.csv       <- IoU/P/R/F1 per labeled image/method
    results/eval_by_condition.csv    <- mean metrics grouped by condition
    results/comparison_table.csv     <- the required 4-column summary table
    results/plots/*.png              <- bar charts
"""

import os
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from segmentation_pipeline import (
    preprocess,
    segment_thresholding,
    segment_region_growing,
    segment_kmeans,
    METHODS,
)

GT_MASK_DIR = "masks"                  # from label_masks.py
LABELS_CSV = "labels.csv"
PRED_MASK_DIR = "results/masks"        # from segmentation_pipeline.py
SEG_RESULTS_CSV = "results/segmentation_results.csv"
IMG_DIR = "dataset"

OUT_DIR = "results"
PLOT_DIR = os.path.join(OUT_DIR, "plots")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def compute_metrics(pred_mask, gt_mask):
    """Pixel-level IoU / Precision / Recall / F1 between two binary masks."""
    pred = (pred_mask > 0)
    gt = (gt_mask > 0)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    union = np.logical_or(pred, gt).sum()

    iou = tp / union if union > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return iou, precision, recall, f1


def load_gt_masks():
    """filename -> ground truth binary mask, for every mask found in masks/."""
    gt = {}
    for path in glob.glob(os.path.join(GT_MASK_DIR, "*_mask.png")):
        base = os.path.basename(path).replace("_mask.png", ".jpg")
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            gt[base] = m
    return gt


# --------------------------------------------------------------------------
# Part 1: evaluate existing predictions against ground truth
# --------------------------------------------------------------------------

def evaluate_predictions(gt_masks, labels_df):
    rows = []
    for fname, gt_mask in gt_masks.items():
        for method in METHODS:
            pred_path = os.path.join(
                PRED_MASK_DIR, method, fname.replace(".jpg", "_mask.png")
            )
            pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            if pred_mask is None:
                print(f"Missing prediction: {pred_path} (run segmentation_pipeline.py first)")
                continue
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(
                    pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            iou, prec, rec, f1 = compute_metrics(pred_mask, gt_mask)
            rows.append({
                "filename": fname, "method": method,
                "iou": iou, "precision": prec, "recall": rec, "f1": f1,
            })

    df = pd.DataFrame(rows)
    if labels_df is not None and not df.empty:
        df = df.merge(labels_df, on="filename", how="left")
    return df


# --------------------------------------------------------------------------
# Part 2: noise / illumination sensitivity test
# --------------------------------------------------------------------------

def add_gaussian_noise(img_bgr, sigma=25):
    noise = np.random.default_rng(0).normal(0, sigma, img_bgr.shape)
    noisy = np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def darken(img_bgr, factor=0.4):
    return np.clip(img_bgr.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def run_method_on_image(method, img_bgr):
    gray, blurred = preprocess(img_bgr)
    if method == "thresholding":
        return segment_thresholding(blurred)
    elif method == "region_growing":
        return segment_region_growing(blurred)
    elif method == "kmeans":
        return segment_kmeans(img_bgr)
    raise ValueError(method)


def sensitivity_test(gt_masks):
    """
    For every labeled image, re-run all three methods on (a) a noisy
    version and (b) a darkened version, and compare IoU to the clean-image
    IoU. The drop in IoU is the "sensitivity" figure for the comparison
    table (bigger drop = more sensitive / less robust).
    """
    rows = []
    for fname, gt_mask in gt_masks.items():
        img_path = os.path.join(IMG_DIR, fname)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Missing source image for sensitivity test: {img_path}")
            continue

        variants = {
            "clean": img_bgr,
            "noisy": add_gaussian_noise(img_bgr),
            "dim": darken(img_bgr),
        }

        for method in METHODS:
            for variant_name, variant_img in variants.items():
                pred_mask = run_method_on_image(method, variant_img)
                iou, prec, rec, f1 = compute_metrics(pred_mask, gt_mask)
                rows.append({
                    "filename": fname, "method": method,
                    "variant": variant_name, "iou": iou,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Part 3: build the required comparison table
# --------------------------------------------------------------------------

def build_comparison_table(eval_df, sens_df, seg_results_df):
    table_rows = []
    for method in METHODS:
        mean_iou = eval_df.loc[eval_df.method == method, "iou"].mean()

        mean_time = np.nan
        if seg_results_df is not None:
            mean_time = seg_results_df.loc[
                seg_results_df.method == method, "processing_time_s"
            ].mean()

        noise_drop = np.nan
        illum_drop = np.nan
        if not sens_df.empty:
            pivot = sens_df[sens_df.method == method].pivot_table(
                index="filename", columns="variant", values="iou"
            )
            if {"clean", "noisy"}.issubset(pivot.columns):
                noise_drop = (pivot["clean"] - pivot["noisy"]).mean()
            if {"clean", "dim"}.issubset(pivot.columns):
                illum_drop = (pivot["clean"] - pivot["dim"]).mean()

        table_rows.append({
            "method": method,
            "segmentation_accuracy_iou": round(mean_iou, 4) if pd.notna(mean_iou) else None,
            "mean_processing_time_s": round(mean_time, 4) if pd.notna(mean_time) else None,
            "iou_drop_under_noise": round(noise_drop, 4) if pd.notna(noise_drop) else None,
            "iou_drop_under_dim_illumination": round(illum_drop, 4) if pd.notna(illum_drop) else None,
        })
    return pd.DataFrame(table_rows)


# --------------------------------------------------------------------------
# Part 4: plots
# --------------------------------------------------------------------------

def plot_metric_by_method(eval_df, metric, title, out_path):
    means = eval_df.groupby("method")[metric].mean().reindex(METHODS)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(means.index, means.values, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_ylim(0, 1)
    for i, v in enumerate(means.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_by_condition(eval_df, condition_col, metric, title, out_path):
    if condition_col not in eval_df.columns:
        return
    pivot = eval_df.pivot_table(
        index=condition_col, columns="method", values=metric, aggfunc="mean"
    ).reindex(columns=METHODS)
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_processing_time(seg_results_df, out_path):
    if seg_results_df is None:
        return
    means = seg_results_df.groupby("method")["processing_time_s"].mean().reindex(METHODS)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(means.index, means.values, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("Mean processing time (s)")
    ax.set_title("Processing Time by Method")
    for i, v in enumerate(means.values):
        ax.text(i, v, f"{v:.3f}s", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sensitivity(sens_df, out_path):
    if sens_df.empty:
        return
    pivot = sens_df.pivot_table(index="method", columns="variant", values="iou", aggfunc="mean")
    pivot = pivot.reindex(index=METHODS, columns=["clean", "noisy", "dim"])
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean IoU")
    ax.set_title("Robustness to Noise / Dim Illumination")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
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
        print(f"No ground truth masks found in {GT_MASK_DIR}/. "
              f"Run label_masks.py first.")
        return
    print(f"Loaded {len(gt_masks)} ground-truth masks.")

    labels_df = pd.read_csv(LABELS_CSV) if os.path.exists(LABELS_CSV) else None
    seg_results_df = (pd.read_csv(SEG_RESULTS_CSV)
                       if os.path.exists(SEG_RESULTS_CSV) else None)

    # Part 1: per-image evaluation against ground truth
    eval_df = evaluate_predictions(gt_masks, labels_df)
    eval_df.to_csv(os.path.join(OUT_DIR, "eval_per_image.csv"), index=False)
    print("Saved results/eval_per_image.csv")

    # Grouped by condition, if labels.csv had condition columns
    condition_cols = [c for c in ["lighting_condition", "background", "touching_objects(y/n)"]
                       if labels_df is not None and c in eval_df.columns]
    if condition_cols:
        by_cond = eval_df.groupby(condition_cols + ["method"])[
            ["iou", "precision", "recall", "f1"]
        ].mean().reset_index()
        by_cond.to_csv(os.path.join(OUT_DIR, "eval_by_condition.csv"), index=False)
        print("Saved results/eval_by_condition.csv")

    # Part 2: sensitivity test (synthetic noise + dim variants)
    print("Running noise/illumination sensitivity test "
          "(re-segmenting noisy + darkened versions of each labeled image)...")
    sens_df = sensitivity_test(gt_masks)
    sens_df.to_csv(os.path.join(OUT_DIR, "sensitivity_results.csv"), index=False)
    print("Saved results/sensitivity_results.csv")

    # Part 3: comparison table
    comparison_df = build_comparison_table(eval_df, sens_df, seg_results_df)
    comparison_df.to_csv(os.path.join(OUT_DIR, "comparison_table.csv"), index=False)
    print("Saved results/comparison_table.csv")
    print(comparison_df.to_string(index=False))

    # Part 4: plots
    plot_metric_by_method(eval_df, "iou", "Mean IoU by Method",
                           os.path.join(PLOT_DIR, "iou_by_method.png"))
    plot_metric_by_method(eval_df, "precision", "Mean Precision by Method",
                           os.path.join(PLOT_DIR, "precision_by_method.png"))
    plot_metric_by_method(eval_df, "recall", "Mean Recall by Method",
                           os.path.join(PLOT_DIR, "recall_by_method.png"))
    plot_metric_by_method(eval_df, "f1", "Mean F1 by Method",
                           os.path.join(PLOT_DIR, "f1_by_method.png"))

    if "lighting_condition" in eval_df.columns:
        plot_metric_by_condition(
            eval_df, "lighting_condition", "iou",
            "IoU by Lighting Condition and Method",
            os.path.join(PLOT_DIR, "iou_by_lighting.png")
        )
    if "touching_objects(y/n)" in eval_df.columns:
        plot_metric_by_condition(
            eval_df, "touching_objects(y/n)", "iou",
            "IoU by Touching vs Non-Touching Coins",
            os.path.join(PLOT_DIR, "iou_by_touching.png")
        )

    plot_processing_time(seg_results_df, os.path.join(PLOT_DIR, "processing_time.png"))
    plot_sensitivity(sens_df, os.path.join(PLOT_DIR, "sensitivity.png"))

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
