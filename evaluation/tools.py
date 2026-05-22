import json
import os

import numpy as np
from labelme import utils
from skimage.io import imread
from skimage.segmentation import relabel_sequential


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def json_to_mask(json_path):
    """Convert a LabelMe JSON annotation to an instance mask."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    height = data.get("imageHeight")
    width = data.get("imageWidth")

    if height is None or width is None:
        raise ValueError(f"{json_path} 缺少 imageHeight 和 imageWidth")

    mask = np.zeros((height, width), dtype=np.int32)
    instance_id = 1
    for shape in data.get("shapes", []):
        if shape["shape_type"] not in ["polygon", "rectangle", "circle", "ellipse"]:
            continue

        points = np.asarray(shape["points"], dtype=np.float32).tolist()
        shape_mask = utils.shape_to_mask((height, width), points, shape_type=shape["shape_type"])
        mask[shape_mask] = instance_id
        instance_id += 1

    return mask


def load_mask_file(file_path):
    """Load a LabelMe JSON or raster mask as an int32 instance mask."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".json":
        return json_to_mask(file_path)
    if ext in [".png", ".tif", ".tiff", ".jpg", ".jpeg"]:
        mask = imread(file_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return np.maximum(mask, 0).astype(np.int32)
    raise ValueError(f"Unsupported file format: {ext}")


def filter_pred_by_gt_overlap_ratio(masks_pred, masks_true, overlap_ratio_threshold: float = 0.3):
    pred_labels = np.unique(masks_pred).astype(np.int32)
    gt_labels = np.unique(masks_true).astype(np.int32)

    pred_ids = pred_labels[pred_labels > 0]  # 真实存在的 pred ID
    gt_ids = gt_labels[gt_labels > 0]  # 真实存在的 gt ID

    n_pred = len(pred_ids)
    n_gt = len(gt_ids)

    if n_pred == 0 or n_gt == 0:
        return np.zeros_like(masks_pred)

    # 重新映射为连续 ID（避免 ID 不连续导致 bincount 出错）
    pred_remap = np.zeros(int(np.max(masks_pred)) + 1, dtype=np.int32)
    gt_remap = np.zeros(int(np.max(masks_true)) + 1, dtype=np.int32)

    pred_remap[pred_ids] = np.arange(1, n_pred + 1)
    gt_remap[gt_ids] = np.arange(1, n_gt + 1)

    pred_remapped = pred_remap[masks_pred]
    gt_remapped = gt_remap[masks_true]

    # 现在 ID 是连续的 1~n，安全使用 bincount
    max_label = max(n_pred, n_gt) + 1
    combined = gt_remapped * max_label + pred_remapped
    combined_flat = combined.ravel()

    counts = np.bincount(combined_flat, minlength=max_label * max_label)
    counts = counts.reshape(max_label, max_label)

    # 交集矩阵：gt x pred
    intersections = counts[1:n_gt + 1, 1:n_pred + 1]  # 关键：只取实际存在的部分

    # 每个 pred 的面积
    pred_areas = np.bincount(pred_remapped.ravel(), minlength=n_pred + 1)[1:]
    # 每个 pred 与所有 gt 的最大交集
    max_intersections = intersections.max(axis=0)  # shape: (n_pred,)

    # 计算比例
    overlap_ratios = max_intersections / pred_areas.astype(float)
    valid = overlap_ratios >= overlap_ratio_threshold

    # 找出要保留的原始 pred ID
    keep_pred_ids = pred_ids[valid]

    # 生成过滤后 mask
    # 1. 使用 np.isin 快速生成包含原始 ID 的过滤后 Mask
    filtered_pred_orig_ids = np.where(np.isin(masks_pred, keep_pred_ids), masks_pred, 0)

    # 2. 使用 relabel_sequential 将非连续的原始 ID 映射为连续的 1, 2, 3...
    filtered_pred, _, _ = relabel_sequential(filtered_pred_orig_ids)

    return filtered_pred
