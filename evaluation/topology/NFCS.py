import argparse
import json
import os
import traceback
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from collections import defaultdict


class CellTopologyEvaluator:
    def __init__(self, gt_path, pred_path, iou_threshold=0.5, structural_threshold=0.5, coord_tolerance=1.0,
                 roi_overlap_threshold=0.5):
        """
        :param structural_threshold: F1-Score 的阈值 (例如 0.5 表示 F1>=0.5 算结构正确)
        :param roi_overlap_threshold: 预测细胞必须有多少比例在GT有效区域内才会被保留
        """
        self.iou_threshold = iou_threshold
        self.structural_threshold = structural_threshold
        self.coord_tolerance = coord_tolerance
        self.roi_overlap_threshold = roi_overlap_threshold

        # 1. 加载原始数据
        # print(f"正在加载 GT 文件: {gt_path}")
        self.gt_data = self._load_data(gt_path)
        # print(f"正在加载 Pred 文件: {pred_path}")
        raw_pred_data = self._load_data(pred_path)

        # --- 过滤不在 GT 区域内的预测细胞 ---
        # print("正在根据 GT 区域过滤预测结果...")
        self.pred_data = self._filter_predictions_by_roi(self.gt_data, raw_pred_data)
        # print(
        #     f"--- 过滤前预测细胞数: {len(raw_pred_data)}   过滤后预测细胞数: {len(self.pred_data)}   GT细胞数:{len(self.gt_data)} ---")

        if len(self.pred_data) == 0:
            print("警告：没有预测细胞落在GT区域内！请检查坐标系是否一致。")

        # 2. 构建图结构
        # print("构建 GT 图结构...")
        self.gt_graph = self._build_graph(self.gt_data)
        # print("构建 Pred 图结构...")
        self.pred_graph = self._build_graph(self.pred_data)

        # 3. 建立映射
        # print("正在匹配 Pred 和 GT 细胞...")
        self.pred_to_gt_map = self._match_cells_by_iou()

    def _load_data(self, path):
        """
        加载数据
        :return:
            cells[cid] = {
                'poly': poly,               # 几何对象(顶点)
                'vertex_count': len(points) # 边数
            }

        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'shapes' in data:
            shapes = data['shapes']
        elif isinstance(data, list):
            shapes = data
        else:
            raise ValueError(f"不支持的JSON格式: {path}")

        cells = {}
        for idx, item in enumerate(shapes):
            cid = str(item["label"])
            points = item.get('points', [])
            if len(points) < 3: continue

            poly = Polygon(points)
            if not poly.is_valid:
                poly = poly.buffer(0)

            # 为了避免重名覆盖，如果label相同，添加后缀
            if cid in cells:
                cid = f"{cid}_{idx}"

            cells[cid] = {
                'poly': poly,
                'vertex_count': len(points)
            }
        return cells

    def _filter_predictions_by_roi(self, gt_cells, pred_cells):
        """
        1. 将所有 GT 细胞合并 (Union) 并外扩一定距离，形成有效区域 (ROI)。
        2. 遍历预测细胞，保留那些与有效区域重叠比例超过阈值的细胞。
        """
        if not gt_cells:
            return pred_cells  # 如果GT为空，无法过滤，返回全部

        # 1. 构建 GT 的有效区域 (ROI)
        gt_polys = [info['poly'] for info in gt_cells.values()]

        # .buffer(20) 的作用是将紧密相邻的细胞融合成一个片区，并向外容错20像素。
        # 如果您希望严格只算GT细胞本体面积，可以将 20 改为 0 或 1。
        gt_roi = unary_union(gt_polys).buffer(0)

        filtered_preds = {}

        # 2. 遍历预测结果进行筛选
        for pid, p_info in pred_cells.items():
            p_poly = p_info['poly']

            # 如果预测细胞本身无效
            if p_poly.area == 0:
                continue

            # 快速检查：如果不相交，直接跳过
            if not p_poly.intersects(gt_roi):
                continue

            # 计算重叠面积
            intersection_area = p_poly.intersection(gt_roi).area
            overlap_ratio = intersection_area / p_poly.area

            # 策略：如果预测细胞有 overlap_ratio 以上的面积落在 GT 区域内，则保留
            if overlap_ratio >= self.roi_overlap_threshold:
                filtered_preds[pid] = p_info

        return filtered_preds

    def _build_graph(self, cells_dict):
        """构建图结构
        :return:
            graph_info[cid] = {
                'neighbors': neighbors, # 邻居label
                'neighbor_count': len(neighbors), # 邻居数量
                'vertex_count': info['vertex_count']    # cid细胞边数
            }
        """
        adj_list = defaultdict(set)
        point_to_ids = defaultdict(list)

        for cid, info in cells_dict.items():
            coords = list(info['poly'].exterior.coords)
            # 去掉重复点，Shapely 会把第一个点再放一遍用于闭合
            if len(coords) > 0 and coords[0] == coords[-1]:
                coords = coords[:-1]
            for pt in coords:
                x_key = float(pt[0])
                y_key = float(pt[1])
                point_to_ids[(x_key, y_key)].append(cid)

                # point_to_ids：坐标+label
                # {
                #     (5.0, 10.0): ["cell_1", "cell_3"],
                #     (8.0, 12.0): ["cell_2"],
                # }

        for pt, ids in point_to_ids.items():
            if len(ids) > 1:
                ids = list(set(ids))  # 去重
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        id_a, id_b = ids[i], ids[j]
                        adj_list[id_a].add(id_b)
                        adj_list[id_b].add(id_a)

        graph_info = {}
        for cid, info in cells_dict.items():
            neighbors = list(adj_list[cid])
            graph_info[cid] = {
                'neighbors': neighbors,
                'neighbor_count': len(neighbors),
                'vertex_count': info['vertex_count']
            }
        return graph_info

    def _match_cells_by_iou(self):
        """建立映射 (保持不变)"""
        mapping = {}
        gt_items = list(self.gt_data.items())

        for pid, p_info in self.pred_data.items():
            best_iou = 0
            best_gid = None
            p_poly = p_info['poly']
            p_bounds = p_poly.bounds

            for gid, g_info in gt_items:
                g_poly = g_info['poly']
                if (p_bounds[2] < g_poly.bounds[0] or p_bounds[0] > g_poly.bounds[2] or
                        p_bounds[3] < g_poly.bounds[1] or p_bounds[1] > g_poly.bounds[3]):
                    continue

                # IoU计算
                if not p_poly.intersects(g_poly):
                    continue
                try:
                    inter = p_poly.intersection(g_poly).area
                    union = p_poly.union(g_poly).area
                    iou = inter / union if union > 0 else 0
                except:
                    iou = 0

                if iou > best_iou:
                    best_iou = iou
                    best_gid = gid

            if best_iou >= self.iou_threshold:
                mapping[pid] = best_gid
            else:
                mapping[pid] = None
        return mapping

    def evaluate(self):
        """
        核心评估 (F1-Score 版)
        Logic:
        1. 遍历每个预测细胞，找到对应的 GT。
        2. 计算 TP: 特征完全匹配的邻居数量。
        3. 计算 Local F1 = (2 * TP) / (N_pred + N_gt)
        4. 如果 F1 >= 阈值，则认为该细胞结构正确。
        5. Global Score = 结构正确的细胞数 / GT 细胞总数
        """
        total_gt_cells = len(self.gt_graph)
        total_pre_cells = len(self.pred_graph)
        if total_gt_cells == 0:
            return {"global_accuracy": 0.0, "details": "No GT cells found"}

        correct_structure_count = 0
        failed_isolated_ids = []
        failed_no_match_ids = []  # 预测细胞没匹配到GT
        failed_score_ids = []  # F1 分数未达标

        # 遍历每一个预测细胞
        for s_id, s_info in self.pred_graph.items():

            # 1. 找到对应的 GT 细胞
            gt_id = self.pred_to_gt_map.get(s_id)

            # 如果该预测细胞没有对应的 GT (False Positive)，则跳过
            if gt_id is None or gt_id not in self.gt_graph:
                failed_no_match_ids.append(s_id)
                continue

            # 2. 获取分母所需的邻居数量
            num_gt_neighbors = self.gt_graph[gt_id]['neighbor_count']
            pred_neighbors = s_info['neighbors']
            num_pred_neighbors = len(pred_neighbors)
            # pre孤立细胞
            if num_pred_neighbors == 0:
                failed_isolated_ids.append(s_id)
                continue
            # 3. 计算分子 (TP: 特征匹配的邻居数)
            correct_neighbor_features_count = 0

            for neighbor_pred_id in pred_neighbors:
                neighbor_gt_id = self.pred_to_gt_map.get(neighbor_pred_id)

                # 如果邻居是假阳性，跳过(pre 无对应 gt)
                if neighbor_gt_id is None or neighbor_gt_id not in self.gt_graph:
                    continue

                # 特征比对
                feat_pred_vertex = self.pred_graph[neighbor_pred_id]['vertex_count']
                feat_pred_n_cnt = self.pred_graph[neighbor_pred_id]['neighbor_count']
                feat_gt_vertex = self.gt_graph[neighbor_gt_id]['vertex_count']
                feat_gt_n_cnt = self.gt_graph[neighbor_gt_id]['neighbor_count']

                # 判定逻辑: 顶点数严格相等(<=0) + 邻居数相等
                vertex_match = abs(feat_pred_vertex - feat_gt_vertex) <= 0
                neighbor_cnt_match = (feat_pred_n_cnt == feat_gt_n_cnt)

                if vertex_match and neighbor_cnt_match:
                    correct_neighbor_features_count += 1

            # 4. 计算 F1 分数 (Dice 系数形式)
            # F1 = 2 * TP / (Precision分母 + Recall分母)
            # F1 = 2 * TP / (|N_pred| + |N_gt|)
            denominator = num_pred_neighbors + num_gt_neighbors

            if denominator > 0:
                f1_score = (2 * correct_neighbor_features_count) / denominator
            else:
                f1_score = 0
                print('error GT存在孤立点')

            # 5. 阈值判定
            if f1_score >= self.structural_threshold:
                correct_structure_count += 1
            else:
                failed_score_ids.append(s_id)

        # 6. 计算 Global Score (基于 GT 总数)
        # global_score = correct_structure_count / total_gt_cells
        denominator = total_gt_cells + total_pre_cells
        if denominator > 0:
            global_score = (2 * correct_structure_count) / denominator
        else:
            global_score = 0.0

        return {
            "global_accuracy": global_score,
            "details": {
                "total_gt_cells": total_gt_cells,
                "total_pred_cells": total_pre_cells,
                "correct_structure_cells": correct_structure_count,
                "failed_isolated_ids": failed_isolated_ids,
                "failed_no_match": failed_no_match_ids,
                "failed_score_ids": failed_score_ids
            }
        }


def batch_evaluate(gt_dir, pred_dir):
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.json')]
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.json')]

    if not gt_files:
        print(f"错误: GT 文件夹 {gt_dir} 为空")
        return

    print(f"开始评估: GT文件 {len(gt_files)} 个, Pred文件 {len(pred_files)} 个")

    all_scores = []

    for gt_fname in gt_files:
        gt_full_path = os.path.join(gt_dir, gt_fname)
        gt_stem = os.path.splitext(gt_fname)[0]

        found_pred_fname = None
        for p_f in pred_files:
            if gt_stem in p_f:
                found_pred_fname = p_f
                break

        if not found_pred_fname:
            print(f"<{gt_fname:<2}>: 未在Pred中找到包含 '{gt_stem}' 的文件")
            continue

        pred_full_path = os.path.join(pred_dir, found_pred_fname)

        try:
            evaluator = CellTopologyEvaluator(
                gt_full_path,
                pred_full_path,
                roi_overlap_threshold=0.5,
                structural_threshold=0.5  # 这里的阈值是针对 F1 Score 的
            )
            res = evaluator.evaluate()

            score = res.get('global_accuracy')
            gt_count = res['details'].get('total_gt_cells')
            pre_count = res['details'].get('total_pred_cells')
            correct = res['details'].get('correct_structure_cells')
            err_iso = len(res['details'].get('failed_isolated_ids', []))
            err_score = len(res['details'].get('failed_score_ids', []))

            all_scores.append(score)

            print(
                f"[{gt_fname:<2}]  "
                f"NFCS(F1版):{score:.4f}  "
                f"GT总数:{gt_count:<3}  "
                f"Pre总数:{pre_count:<3}  "
                f"结构正确:{correct:<3} "
                f"结构错误:{err_iso + err_score:<3} "
                f"(孤立细胞:{err_iso} | 未达阈值:{err_score})"
            )

        except Exception as e:
            print(f"{gt_fname:<25} | 错误: {str(e)}")
            # traceback.print_exc()

    if all_scores:
        avg = np.mean(all_scores)
        print(f"【汇总报告】 平均细胞拓扑准确度 (mNFCS): {avg:.4f} (基于 {len(all_scores)} 个文件)")
    else:
        print("未成功评估任何文件。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cell neighborhood topology with NFCS.")
    parser.add_argument("--gt_dir", required=True, help="Ground-truth JSON directory.")
    parser.add_argument("--pred_dir", required=True, help="Prediction JSON directory.")
    args = parser.parse_args()

    try:
        if not os.path.exists(args.gt_dir) or not os.path.exists(args.pred_dir):
            print("错误: 文件夹路径不存在！")
        else:
            batch_evaluate(args.gt_dir, args.pred_dir)
    except Exception as e:
        traceback.print_exc()
