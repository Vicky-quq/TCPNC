import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["pdf.fonttype"] = 42

def plot_dataset_axes(fig, grid_pos, errors_list, aps_list, names, cols, overall_errors, overall_aps, d_name, prefix):
    subgrid = grid_pos.subgridspec(1, 2, wspace=0.3)
    # pos = grid_pos.get_position(fig)
    fig.text(
        0.1 if prefix == "(A)" else 0.52, 0.95,
        f"{prefix} Evaluation on the {d_name}",
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="top"
    )

    ax1 = fig.add_subplot(subgrid[0, 0])
    for i in range(len(errors_list)):
        vp = ax1.violinplot(errors_list[i], positions=[i], widths=0.8, showmeans=False, showextrema=False)
        vp["bodies"][0].set_facecolor(cols[i])
        vp["bodies"][0].set_alpha(0.4)
        ax1.plot(0.3 * np.array([-1, 1]) + i, overall_errors[i] * np.ones(2), color=cols[i], lw=3)
        ax1.text(i, overall_errors[i], f"{overall_errors[i]:.3f}", color="black", 
                 fontsize=10, fontweight="bold", ha="center", va="bottom")
        x_jitter = np.random.normal(i, 0.04, size=len(errors_list[i]))
        ax1.scatter(x_jitter, errors_list[i], color=cols[i], s=15, alpha=0.7, edgecolors="none")

    ax1.set_ylabel("error rate @ 0.5 IoU", fontsize=12)
    ax1.text(0.38, 1.05, r"= $\frac{FP + FN}{TP + FN}$", transform=ax1.transAxes, fontsize=16)
    ax1.text(0, 1.05, r"error rate", transform=ax1.transAxes, fontsize=12,ha="left", va="bottom")
    ax1.set_ylim([0, 0.2])
    ax1.set_yticks(np.arange(0, 0.21, 0.05))
    ax1.tick_params(axis="both", labelsize=12)
    ax1.set_xticks(np.arange(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha="right")
    for i, t in enumerate(ax1.get_xticklabels()):
        t.set_color(cols[i])
        t.set_weight("bold")

    ax2 = fig.add_subplot(subgrid[0, 1])
    for i in range(len(aps_list)):
        vp = ax2.violinplot(aps_list[i], positions=[i], widths=0.8, showmeans=False, showextrema=False)
        vp["bodies"][0].set_facecolor(cols[i])
        vp["bodies"][0].set_alpha(0.4)
        ax2.plot(0.3 * np.array([-1, 1]) + i, overall_aps[i] * np.ones(2), color=cols[i], lw=3)
        # 使用偏移量确保从短横线下边缘开始
        offset = 0.002  # 可调
        ax2.text(i, overall_aps[i] - offset,
                 f"{overall_aps[i]:.3f}",
                 color="black",
                 fontsize=10,
                 fontweight="bold",
                 ha="center",
                 va="top")
        x_jitter = np.random.normal(i, 0.04, size=len(aps_list[i]))
        ax2.scatter(x_jitter, aps_list[i], color=cols[i], s=15, alpha=0.7, edgecolors="none")

    ax2.set_ylabel("average precision (AP) @ 0.5 IoU", fontsize=12)
    ax2.text(0.38, 1.05, r"= $\frac{TP}{TP + FN + FP}$", transform=ax2.transAxes, fontsize=16)
    ax2.text(0, 1.07, "average\nprecision", transform=ax2.transAxes, fontsize="large", va="center")
    ax2.set_ylim([0.80, 1.0])
    ax2.set_yticks(np.arange(0.80, 1.01, 0.05))
    ax2.tick_params(axis="both", labelsize=12)
    ax2.set_xticks(np.arange(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right")
    for i, t in enumerate(ax2.get_xticklabels()):
        t.set_color(cols[i])
        t.set_weight("bold")

def draw_violin_plot(result_dir, output_dir=None, datasets=None, algs=None, alg_names=None):
    output_dir = output_dir or result_dir
    os.makedirs(output_dir, exist_ok=True)

    datasets = datasets or ["Sansha-5", "WO115-2"]
    algs = algs or ["cyto3", "cyto3_ours", "cellposesam", "cellposesam_ours"]
    alg_names = alg_names or ["cyto3", "cyto3_Ours", "Cellpose-SAM", "Cellpose-SAM_Ours"]
    colors = ["#6F669B", "#5D99BB", "#5F8E84", "#BC7365"]

    fig = plt.figure(figsize=(15, 7))
    master_grid = plt.GridSpec(1, len(datasets), wspace=0.2, bottom=0.2, top=0.85)

    for idx, dataset_name in enumerate(datasets):
        prefix = f"({chr(ord('A') + idx)})"
        aps_data = []
        errors_data = []
        overall_errors = []
        overall_aps = []
        used_alg_names = []
        used_colors = []

        for alg_idx, alg in enumerate(algs):
            file_path = os.path.join(result_dir, f"{alg}_{dataset_name}.npy")
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            dat = np.load(file_path, allow_pickle=True).item()
            tp, fp, fn = dat["tp"][:, 0], dat["fp"][:, 0], dat["fn"][:, 0]
            aps_data.append(dat["ap"][:, 0])
            errors_data.append((fp + fn) / (tp + fn))

            total_tp, total_fp, total_fn = tp.sum(), fp.sum(), fn.sum()
            overall_errors.append((total_fp + total_fn) / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0)
            overall_aps.append(total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0)
            used_alg_names.append(alg_names[alg_idx] if alg_idx < len(alg_names) else alg)
            used_colors.append(colors[alg_idx % len(colors)])

        if len(aps_data) > 0:
            plot_dataset_axes(
                fig,
                master_grid[0, idx],
                errors_data,
                aps_data,
                used_alg_names,
                used_colors,
                overall_errors,
                overall_aps,
                dataset_name,
                prefix
            )

    save_path = os.path.join(output_dir, "result_segmentation.pdf")
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Combined figure saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw segmentation AP/error violin plots from NPY files.")
    parser.add_argument("--result_dir", required=True, help="Directory containing *_Dataset.npy files.")
    parser.add_argument("--output_dir", default=None, help="Directory for result_segmentation.pdf.")
    parser.add_argument("--datasets", nargs="+", default=["Sansha-5", "WO115-2"],
                        help="Dataset names matching the NPY filename suffix.")
    parser.add_argument("--algs", nargs="+", default=["cyto3", "cyto3_ours", "cellposesam", "cellposesam_ours"],
                        help="Algorithm names matching the NPY filename prefix.")
    parser.add_argument("--alg_names", nargs="+", default=["cyto3", "cyto3_Ours", "Cellpose-SAM", "Cellpose-SAM_Ours"],
                        help="Display names for algorithms.")
    args = parser.parse_args()

    draw_violin_plot(args.result_dir, args.output_dir, args.datasets, args.algs, args.alg_names)
