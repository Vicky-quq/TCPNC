import argparse
import os

import numpy as np
from cellpose4 import utils

from evaluation import tools


def load_mask_file(file_path):
    """读取 json 或 mask 图像"""
    return tools.load_mask_file(file_path)


def compute_mean_diameter(gt_root, dataset_names):
    for dataset_name in dataset_names:

        gt_dataset_dir = os.path.join(gt_root, dataset_name)

        if not os.path.isdir(gt_dataset_dir):
            print(f"[SKIP] {dataset_name} not found")
            continue

        print(f"\nDataset: {dataset_name}")

        gt_files = [f for f in os.listdir(gt_dataset_dir) if f.lower().endswith(".json")]
        gt_files.sort()

        all_diams = []

        for gt_file in gt_files:

            gt_path = os.path.join(gt_dataset_dir, gt_file)

            masks_gt = load_mask_file(gt_path)

            diam = utils.diameters(masks_gt)[0]

            all_diams.append(diam)

        if len(all_diams) > 0:

            avg_diam = np.mean(all_diams)
            median_diam = np.median(all_diams)

            print(f"Images: {len(all_diams)}")
            print(f"Mean cell diameter : {avg_diam:.2f} pixels")
            print(f"Median diameter    : {median_diam:.2f} pixels")

        else:
            print("No GT masks found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mean/median cell diameter for GT datasets.")
    parser.add_argument("--gt_root", required=True, help="Root directory containing dataset GT folders.")
    parser.add_argument("--datasets", nargs="+", default=["WO115-2", "Sansha-5"],
                        help="Dataset folder names to process.")
    args = parser.parse_args()

    compute_mean_diameter(args.gt_root, args.datasets)
