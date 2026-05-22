import argparse
import os

import numpy as np
import tifffile
from cellpose4 import metrics

from evaluation import tools


def evaluate_dataset(pred_root, gt_root, output_dir=None, overlap_threshold=0.3):
    dset_name = os.path.basename(os.path.normpath(pred_root))
    save_root = output_dir or os.path.dirname(os.path.normpath(pred_root))
    tools.ensure_dir(save_root)

    gt_files = [f for f in os.listdir(gt_root) if f.endswith(".json")]

    for model_name in os.listdir(pred_root):
        model_dir = os.path.join(pred_root, model_name)
        if not os.path.isdir(model_dir):
            continue

        print(f"\nProcessing model: {model_name}")

        masks_true = []
        masks_pred = []

        pred_files = [f for f in os.listdir(model_dir) if f.lower().endswith((".json", ".tif", ".tiff"))]

        processed_filenames = []
        for pred_file in pred_files:
            pred_name = os.path.splitext(pred_file)[0]

            matched_gt = next((f for f in gt_files if os.path.splitext(f)[0] in pred_name), None)

            if matched_gt is None:
                continue

            gt_path = os.path.join(gt_root, matched_gt)
            gt_mask = tools.json_to_mask(gt_path)

            pred_path = os.path.join(model_dir, pred_file)
            if pred_file.lower().endswith(".json"):
                pred_mask = tools.json_to_mask(pred_path)
            else:
                pred_mask = tifffile.imread(pred_path)

            pred_mask = tools.filter_pred_by_gt_overlap_ratio(
                pred_mask,
                gt_mask,
                overlap_ratio_threshold=overlap_threshold
            )

            masks_true.append(gt_mask)
            masks_pred.append(pred_mask)
            processed_filenames.append(pred_file)

        if len(masks_true) == 0:
            print(f"No matched files for {model_name}")
            continue

        threshold = np.array([0.5, 0.75, 0.9])
        ap, tp, fp, fn = metrics.average_precision(
            masks_true,
            masks_pred,
            threshold=threshold
        )

        save_path = os.path.join(
            save_root,
            f"{model_name}_{dset_name}.npy"
        )

        np.save(save_path, {
            "ap": ap,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "threshold": threshold,
            "filenames": processed_filenames,
            "masks_true": masks_true,
            "masks_pred": masks_pred
        })

        print(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert segmentation evaluation results to NPY files.")
    parser.add_argument("--pred_root", required=True, help="Prediction root containing one subfolder per model.")
    parser.add_argument("--gt_root", required=True, help="Ground-truth JSON directory.")
    parser.add_argument("--output_dir", default=None, help="Directory for generated NPY files.")
    parser.add_argument("--overlap_threshold", type=float, default=0.3,
                        help="Minimum predicted-instance overlap ratio inside the GT ROI.")
    args = parser.parse_args()

    evaluate_dataset(args.pred_root, args.gt_root, args.output_dir, args.overlap_threshold)
