import matplotlib
from matplotlib.ticker import MaxNLocator

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["pdf.fonttype"] = 42

import os
import json
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import entropy, ks_2samp
from shapely.geometry import Polygon
from shapely.ops import unary_union


class ROIFilter:
    def __init__(self, roi_overlap_threshold=0.5):
        self.roi_overlap_threshold = roi_overlap_threshold

    def _filter_predictions_by_roi(self, gt_cells, pred_cells):
        if not gt_cells:
            return pred_cells
        gt_polys = [info["poly"] for info in gt_cells.values()]
        gt_roi = unary_union(gt_polys).buffer(0)
        filtered_preds = {}
        for pid, p_info in pred_cells.items():
            p_poly = p_info["poly"]
            if p_poly.area == 0:
                continue
            if not p_poly.intersects(gt_roi):
                continue
            intersection_area = p_poly.intersection(gt_roi).area
            overlap_ratio = intersection_area / p_poly.area
            if overlap_ratio >= self.roi_overlap_threshold:
                filtered_preds[pid] = p_info
        return filtered_preds


def collect_json_files(folder):
    return [os.path.join(root, f)
            for root, _, fs in os.walk(folder)
            for f in fs if f.endswith(".json")]

def load_shapes(json_path):
    with open(json_path) as f:
        data = json.load(f)
    shapes = data.get("shapes", [])
    cells = {}
    for idx, s in enumerate(shapes):
        pts = s.get("points", [])
        if len(pts) >= 3:
            poly = Polygon(pts)
            cells[idx] = {"poly": poly}
    return cells

def analyze_points(gt_folder, pred_folder, roi_threshold=0.5):
    gt_files = collect_json_files(gt_folder)
    pred_files = collect_json_files(pred_folder)
    roi_filter = ROIFilter(roi_overlap_threshold=roi_threshold)
    points_list = []
    distribution = defaultdict(int)

    for gt_file in gt_files:
        gt_name = os.path.splitext(os.path.basename(gt_file))[0]
        matched_pred_files = [pf for pf in pred_files if gt_name in os.path.splitext(os.path.basename(pf))[0]]
        if not matched_pred_files:
            continue
        gt_cells = load_shapes(gt_file)
        for pred_file in matched_pred_files:
            pred_cells = load_shapes(pred_file)
            filtered_pred_cells = roi_filter._filter_predictions_by_roi(gt_cells, pred_cells)
            for cell in filtered_pred_cells.values():
                n_points = len(cell["poly"].exterior.coords) - 1
                distribution[n_points] += 1
                points_list.append(n_points)
    return distribution, points_list

def compute_statistics(gt_points, pred_points):
    gt_hist = np.bincount(gt_points)
    pred_hist = np.bincount(pred_points)
    max_len = max(len(gt_hist), len(pred_hist))
    gt_hist = np.pad(gt_hist, (0, max_len - len(gt_hist)))
    pred_hist = np.pad(pred_hist, (0, max_len - len(pred_hist)))
    gt_prob = gt_hist / np.sum(gt_hist)
    pred_prob = pred_hist / np.sum(pred_hist)
    kl = entropy(gt_prob + 1e-10, pred_prob + 1e-10)
    ks_stat, ks_p = ks_2samp(gt_points, pred_points)
    return kl, ks_stat, ks_p

def plot_dataset_row(fig, grid_pos, gt_points, pred_points_dict, dataset_name, prefix):
    if not gt_points:
        return

    all_points_lists = [gt_points] + list(pred_points_dict.values())
    titles = ["Ground Truth"] + list(pred_points_dict.keys())

    global_max_density = 0
    all_hist_data = []
    for points in all_points_lists:
        if not points or len(points) == 0:
            all_hist_data.append((None, None, 0))
            continue
        hist, bin_edges = np.histogram(points, bins=np.arange(min(points), max(points) + 2), density=True)
        global_max_density = max(global_max_density, hist.max())
        all_hist_data.append((hist, bin_edges, hist.max()))

    n_plots = len(all_points_lists)
    subgrid = grid_pos.subgridspec(1, n_plots, wspace=0.1)
    
    pos = grid_pos.get_position(fig)
    fig.text(
        0.1, pos.y1 + 0.04,
        f"{prefix} Evaluation on {dataset_name}",
        fontsize=14, fontweight="bold", fontfamily="Arial", ha="left", va="bottom"
    )

    colors = ["gray"] + ["#BC7365", "#5D99BB", "#6E9C74", "#9E8BB4"]
    color_cycle = [colors[i % len(colors)] for i in range(n_plots)]

    for i in range(n_plots):
        ax = fig.add_subplot(subgrid[0, i])
        points = all_points_lists[i]
        title = titles[i]
        color = color_cycle[i]

        if points is None or len(points) == 0:
            ax.set_title(title + "\n(no data)", fontsize=12, fontfamily="Arial")
            continue

        hist, bin_edges, _ = all_hist_data[i]
        ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges),
               align="edge", alpha=0.8, color=color, edgecolor="black")

        ax.set_xlabel("Vertices per Cell", fontsize=12, fontfamily="Arial")
        ax.set_title(title, fontsize=12, fontfamily="Arial", fontweight="bold")
        ax.set_ylim(0, global_max_density * 1.1)
        ax.tick_params(axis="both", labelsize=10)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontfamily("Arial")

        if i == 0:
            ax.set_ylabel("Proportion", fontsize=12, fontfamily="Arial")
        else:
            ax.yaxis.set_tick_params(labelleft=False)
            
        if i == n_plots - 1:
            ax_count = ax.twinx()
            total_count = len(points)
            ax_count.set_ylim(ax.get_ylim()[0] * total_count, ax.get_ylim()[1] * total_count)
            ax_count.set_ylabel("Count", fontsize=12, fontfamily="Arial")
            ax_count.yaxis.set_major_locator(MaxNLocator(nbins=5))
            for label in ax_count.get_yticklabels():
                label.set_fontfamily("Arial")
            ax_count.tick_params(axis="y", labelsize=10)

def run_all_evaluations(pred_root, gt_root, roi_threshold=0.5, datasets=None, target_algs=None):
    os.makedirs(pred_root, exist_ok=True)
    datasets = datasets or [
        {"name": "Sansha-5", "prefix": "(A)"},
        {"name": "WO115-2", "prefix": "(B)"}
    ]
    target_algs = target_algs or ["cellposesam_ours", "cyto3_ours"]

    # fig = plt.figure(figsize=(14, 10))
    # master_grid = plt.GridSpec(2, 1, hspace=0.4, left=0.1, right=0.9, bottom=0.1, top=0.9)

    for idx, dataset in enumerate(datasets):
        dataset_name = dataset["name"]
        prefix = dataset["prefix"]
        gt_folder = os.path.join(gt_root, dataset_name)
        pred_base_folder = os.path.join(pred_root, dataset_name)
        
        if not os.path.exists(gt_folder): continue

        pred_folders = []
        for alg in target_algs:
            alg_path = os.path.join(pred_base_folder, alg)
            if os.path.exists(alg_path): 
                pred_folders.append(alg_path)

        if not pred_folders: continue

        # gt_dist: {n_points: count}, gt_points: [n_points, ...]
        gt_dist, gt_points = analyze_points(gt_folder, gt_folder, roi_threshold)
        total_gt = len(gt_points)
        
        all_pred_points = {}
        all_pred_dists = {}
        for pred_folder in pred_folders:
            alg_name = os.path.basename(os.path.normpath(pred_folder))
            p_dist, p_points = analyze_points(gt_folder, pred_folder, roi_threshold)
            all_pred_points[alg_name] = p_points
            all_pred_dists[alg_name] = p_dist
        
        # plot_dataset_row(fig, master_grid[idx, 0], gt_points, all_pred_points, dataset_name, prefix)

        # Generate CSV data
        all_keys = set(gt_dist.keys())
        for dist in all_pred_dists.values():
            all_keys.update(dist.keys())
        
        sorted_keys = sorted(list(all_keys))
        
        rows = []
        # Header Row 1: Vertice=X, ..., total num
        # First column: Method (GT, cellpose_ours, cyto3_ours)
        
        def get_row_data(label, dist, total):
            row = {"Method": label}
            for k in sorted_keys:
                count = dist.get(k, 0)
                percentage = (count / total * 100) if total > 0 else 0
                # Using the format: 30%(125)
                row[f"Vertice={k}"] = f"{percentage:.1f}%({count})"
            row["total num"] = total
            return row

        csv_rows = []
        csv_rows.append(get_row_data("GT", gt_dist, total_gt))
        
        name_map = {
            "cellposesam_ours": "cellposesam_ours",
            "cyto3_ours": "cyto3_ours"
        }
        
        for alg_name, p_dist in all_pred_dists.items():
            display_name = name_map.get(alg_name, alg_name)
            csv_rows.append(get_row_data(display_name, p_dist, len(all_pred_points[alg_name])))
            
        df = pd.DataFrame(csv_rows)
        # Reorder columns: Method, Vertice=3, Vertice=4, ..., total num
        cols = ["Method"] + [f"Vertice={k}" for k in sorted_keys] + ["total num"]
        df = df[cols]
        
        csv_filename = os.path.join(pred_root, f"vertex_analysis_{dataset_name}.csv")
        # To make the header more like the image (Top-left cell empty), we can rename 'Method' to ''
        df.rename(columns={"Method": ""}, inplace=True)
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"CSV saved to {csv_filename}")

    # save_path = os.path.join(root, "result_VertexStatistics.pdf")
    # plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    # plt.close(fig)
    # print(f"Results saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze polygon vertex-count distributions.")
    parser.add_argument("--pred_root", "--root", dest="pred_root", required=True,
                        help="Prediction result root containing dataset folders.")
    parser.add_argument("--gt_root", required=True, help="Ground-truth root containing dataset folders.")
    parser.add_argument("--datasets", nargs="+", default=["Sansha-5", "WO115-2"],
                        help="Dataset folder names to evaluate.")
    parser.add_argument("--algs", nargs="+", default=["cellposesam_ours", "cyto3_ours"],
                        help="Prediction method subfolders to evaluate.")
    parser.add_argument("--roi_threshold", type=float, default=0.5)
    args = parser.parse_args()
    dataset_configs = [{"name": name, "prefix": f"({chr(ord('A') + idx)})"} for idx, name in enumerate(args.datasets)]
    run_all_evaluations(args.pred_root, args.gt_root, args.roi_threshold, dataset_configs, args.algs)
