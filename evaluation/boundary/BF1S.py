import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from cellpose4 import metrics

from evaluation import tools

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['pdf.fonttype'] = 42


def load_mask_file(file_path):
    """统一读取 json 或 mask 图像"""
    return tools.load_mask_file(file_path)


def run_dataset_evaluation(dataset_name, avg_diam, scales, title_prefix, gt_root, pred_root, output_dir, algs):
    gt_dir = os.path.join(gt_root, dataset_name)
    pred_dir = os.path.join(pred_root, dataset_name)
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith('.json')])

    dataset_plot_data = {}
    txt_path = os.path.join(output_dir, f"boundary_evaluation_{dataset_name}.txt")

    with open(txt_path, 'w', encoding='utf-8') as f_out:
        f_out.write(f"Dataset: {dataset_name}\nAvg Diam: {avg_diam}\n\n")

        for alg in algs:
            model_path = os.path.join(pred_dir, alg)
            if not os.path.isdir(model_path): continue

            # 1. 收集当前算法的所有 Mask 列表
            list_gt = []
            list_pred_filtered = []

            for gf in gt_files:
                gt_name = os.path.splitext(gf)[0]
                pf = next((f for f in os.listdir(model_path) if gt_name in f), None)
                if pf:
                    m_gt = load_mask_file(os.path.join(gt_dir, gf))
                    m_pred = load_mask_file(os.path.join(model_path, pf))
                    m_filt = tools.filter_pred_by_gt_overlap_ratio(m_pred, m_gt, 0.5)

                    list_gt.append(m_gt)
                    list_pred_filtered.append(m_filt)

            # 2. 一次性传入列表计算
            if len(list_gt) > 0:
                print(f"[{dataset_name}] Evaluating {alg} with {len(list_gt)} images...")
                prec, rec, f1 = metrics.boundary_scores(list_gt, list_pred_filtered, scales)

                mean_prec = prec.mean(axis=1)
                mean_rec = rec.mean(axis=1)
                mean_f1 = f1.mean(axis=1)

                dataset_plot_data[alg] = {'prec': mean_prec, 'rec': mean_rec, 'f1': mean_f1}

                f_out.write(f"Model: {alg} (Images: {len(list_gt)})\n")
                f_out.write(f"Scale-F1: {mean_f1}\n\n")

    if dataset_plot_data:
        draw_plot(dataset_name, dataset_plot_data, scales, avg_diam, title_prefix, output_dir, algs)


def draw_plot(dataset_name, model_data, scales_val, avg_diam, prefix, output_dir, algs):
    metrics_names = ['prec', 'rec', 'f1']
    titles = ['Boundary Precision', 'Boundary Recall', 'Boundary F-score']
    colors = ["#6F669B", "#5D99BB", "#5F8E84", "#BC7365"]
    x_pixels = scales_val * avg_diam

    display_names = {
        "cyto3": "cyto3",
        "cyto3_ours": "cyto3_Ours ",
        "cellposesam": "Cellpose-SAM",
        "cellposesam_ours": "Cellpose-SAM_Ours"
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300, facecolor='w')

    for i, m_name in enumerate(metrics_names):
        ax = axes[i]
        for idx, alg in enumerate(algs):
            if alg in model_data:
                label_name = display_names.get(alg, alg)
                ax.plot(x_pixels, model_data[alg][m_name], color=colors[idx % len(colors)],
                        label=label_name, marker='o', markersize=3, lw=1.2)

        ax.set_title(titles[i], fontsize=14, fontweight='bold', fontfamily='Arial')
        ax.set_xlabel('boundary width (pixels)', fontsize=14, fontweight='bold', fontfamily='Arial')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold', fontfamily='Arial')
        ax.set_ylim(0.3, 1.0)
        # ax.set_xlim(0, 8.5)
        # # x_limit = np.ceil(max(x_pixels) * 2) / 2
        # # ax.set_xlim(0, x_limit)
        # # ax.xaxis.set_major_locator(MultipleLocator(0.5))
        # # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:g}"))

        ax.tick_params(axis='both', labelsize=14)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontfamily('Arial')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        if i == 0: ax.legend(loc='lower right', frameon=False, fontsize=12)

    fig.text(0.02, 0.99, f"{prefix} Evaluation on the {dataset_name}",
             fontsize=14, fontweight='bold', fontfamily='Arial', ha="left", va="top")

    save_path = os.path.join(output_dir, f"boundary_score_{dataset_name}.pdf")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved plot: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate boundary precision, recall and F-score.")
    parser.add_argument("--gt_root", required=True, help="Root directory containing dataset GT folders.")
    parser.add_argument("--pred_root", required=True, help="Root directory containing dataset prediction folders.")
    parser.add_argument("--output_dir", default=None, help="Directory for TXT and PDF outputs.")
    parser.add_argument("--algs", nargs="+", default=["cyto3", "cyto3_ours", "cellposesam", "cellposesam_ours"],
                        help="Prediction method subfolders to evaluate.")
    args = parser.parse_args()

    output_dir = args.output_dir or args.pred_root
    tools.ensure_dir(output_dir)

    datasets = [
        {"name": "Sansha-5", "avg_diam": 147, "scales": np.arange(0.005, 0.060, 0.005), "prefix": "(A)"},
        {"name": "WO115-2", "avg_diam": 122, "scales": np.arange(0.006, 0.070, 0.006), "prefix": "(B)"},
    ]
    for config in datasets:
        run_dataset_evaluation(
            config["name"],
            config["avg_diam"],
            config["scales"],
            config["prefix"],
            args.gt_root,
            args.pred_root,
            output_dir,
            args.algs,
        )
    print("\n[ALL DONE]")
