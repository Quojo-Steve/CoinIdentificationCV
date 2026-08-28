"""
Ground-truth mask labeling tool for the coin segmentation project.

    pip install opencv-python matplotlib numpy
    python label_masks.py

Run this from the folder that contains your 'dataset/' folder
(the one with coin_001.jpg ... coin_023.jpg), or edit IMG_DIR below.

HOW TO USE (per coin, per image):
    - Left-click points around one coin's edge (roughly following its boundary,
      6-10 points is plenty for a coin).
    - Press ENTER (or middle-click) to close that coin's outline.
    - Right-click to undo the last point if you misclick.
    - After closing a coin's outline, the window updates showing your progress
      in green. Keep clicking to outline the next coin.
    - When you've outlined every coin in the image, just CLOSE the window
      (the X button) to move to the next image.

The mask for each image is saved as a binary PNG (white = coin, black =
background) in the 'masks/' folder, ready for IoU / Precision / Recall later.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

IMG_DIR = "dataset"
MASK_DIR = "masks"

# Representative subset spanning lighting, background, count, and touching conditions.
# Edit this list if you'd rather label a different set of images.
SELECTED = [
    "coin_001.jpg",  # 8 coins, tile, bright, touching
    "coin_002.jpg",  # 8 coins, tile, shadow, touching
    "coin_006.jpg",  # 2 coins, brown wood, bright, not touching (easy case)
    "coin_010.jpg",  # 1 coin, light wood, dim (easiest case)
    "coin_012.jpg",  # 6 coins, light wood, blurry, touching
    "coin_017.jpg",  # 6 coins, light wood, bright, not touching
    "coin_018.jpg",  # 6 coins, light wood, shadow, not touching
    "coin_022.jpg",  # 6 coins, light wood, dim, not touching
]


def label_image(path):
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    coin_count = 0

    fig, ax = plt.subplots(figsize=(8, 8 * h / w))
    ax.imshow(img_rgb)
    ax.set_title(
        "Click around a coin's edge, ENTER to close it.\n"
        "Right-click = undo last point. Close window when all coins are done."
    )
    plt.tight_layout()
    fig.canvas.draw()

    while True:
        pts = plt.ginput(n=-1, timeout=0)
        if plt.fignum_exists(fig.number) is False:
            break  # window was closed
        if len(pts) < 3:
            continue  # not enough points for a polygon, try again
        poly = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)
        coin_count += 1

        overlay = img_rgb.copy()
        overlay[mask > 0] = (0, 255, 0)
        blended = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
        ax.clear()
        ax.imshow(blended)
        ax.set_title(
            f"{coin_count} coin(s) outlined so far.\n"
            "Click next coin, or close window when finished."
        )
        fig.canvas.draw()

    plt.close(fig)
    return mask, coin_count


def main():
    os.makedirs(MASK_DIR, exist_ok=True)
    for fname in SELECTED:
        path = os.path.join(IMG_DIR, fname)
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue
        print(f"\nLabeling {fname} -- outline each coin, close window when done.")
        mask, n = label_image(path)
        out_path = os.path.join(MASK_DIR, fname.replace(".jpg", "_mask.png"))
        cv2.imwrite(out_path, mask)
        print(f"Saved mask with {n} coin(s) -> {out_path}")

    print("\nAll done. Ground-truth masks saved in:", MASK_DIR)


if __name__ == "__main__":
    main()
