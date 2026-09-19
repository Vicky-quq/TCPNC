import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import json
import os
import argparse
from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries
from scipy import ndimage as ndi
from collections import Counter, defaultdict


def get_node_color(deg):
    """Return RGB color based on node degree."""
    if deg == 1: return 0, 255, 0
    if deg == 2: return 255, 255, 0
    if deg == 3: return 0, 0, 0
    if deg >= 4: return 0, 0, 255
    return 128, 128, 128


def optimize_four_way_junctions(G, final_polygons_points, refined_centroids, split_ratio=0.1):
    """Split 4-way junctions based on topology edge association and sector matching."""

    def to_t(pt):
        return (round(float(pt[0]), 2), round(float(pt[1]), 2))

    edge_to_polys = {}
    for p_idx, poly in enumerate(final_polygons_points):
        n = len(poly)
        for i in range(n):
            p1_t = to_t(poly[i])
            p2_t = to_t(poly[(i + 1) % n])
            edge = tuple(sorted([p1_t, p2_t]))
            if edge not in edge_to_polys:
                edge_to_polys[edge] = []
            edge_to_polys[edge].append(p_idx)

    modified_polygons = [[list(pt) for pt in poly] for poly in final_polygons_points]
    four_way_nodes = [n for n, d in G.degree() if d == 4]

    optimization_debug_lines = []

    for node_idx in four_way_nodes:
        v_coords = np.array(refined_centroids[node_idx])
        v_t = to_t(v_coords)

        neighbors_idx = list(G.neighbors(node_idx))
        neighbors_idx.sort(key=lambda ni: np.arctan2(refined_centroids[ni][1] - v_coords[1],
                                                     refined_centroids[ni][0] - v_coords[0]))

        is_outer_boundary = False
        edge_polys_sets = []
        for n_idx in neighbors_idx:
            n_t = to_t(refined_centroids[n_idx])
            edge = tuple(sorted([v_t, n_t]))
            polys = edge_to_polys.get(edge, [])
            if len(polys) < 2:
                is_outer_boundary = True
                break
            edge_polys_sets.append(set(polys))

        if is_outer_boundary:
            continue

        cell_indices = []
        for i in range(4):
            common = edge_polys_sets[i].intersection(edge_polys_sets[(i + 1) % 4])
            if not common:
                break
            cell_indices.append(list(common)[0])

        if len(cell_indices) < 4:
            continue

        sum02 = len(modified_polygons[cell_indices[0]]) + len(modified_polygons[cell_indices[2]])
        sum13 = len(modified_polygons[cell_indices[1]]) + len(modified_polygons[cell_indices[3]])

        poly_centers = [np.mean(modified_polygons[idx], axis=0) for idx in cell_indices]

        def poly_area(poly):
            x = np.array([p[0] for p in poly])
            y = np.array([p[1] for p in poly])
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        use_13 = False
        if sum13 > sum02:
            use_13 = True
        elif sum13 < sum02:
            use_13 = False
        else:
            area13 = poly_area(modified_polygons[cell_indices[1]]) + poly_area(modified_polygons[cell_indices[3]])
            area02 = poly_area(modified_polygons[cell_indices[0]]) + poly_area(modified_polygons[cell_indices[2]])
            if area13 >= area02:
                use_13 = True
            else:
                use_13 = False

        if use_13:
            p_to_separate = [cell_indices[1], cell_indices[3]]
            p_to_add_edge = [cell_indices[0], cell_indices[2]]
            dir1 = poly_centers[1] - v_coords
            dir2 = poly_centers[3] - v_coords
            optimization_debug_lines.append((poly_centers[1].tolist(), poly_centers[3].tolist()))
        else:
            p_to_separate = [cell_indices[0], cell_indices[2]]
            p_to_add_edge = [cell_indices[1], cell_indices[3]]
            dir1 = poly_centers[0] - v_coords
            dir2 = poly_centers[2] - v_coords
            optimization_debug_lines.append((poly_centers[0].tolist(), poly_centers[2].tolist()))

        avg_branch_len = np.mean([np.linalg.norm(refined_centroids[ni] - v_coords) for ni in neighbors_idx])
        mag = avg_branch_len * split_ratio

        v1_new = (v_coords + (dir1 / (np.linalg.norm(dir1) + 1e-6)) * mag).tolist()
        v2_new = (v_coords + (dir2 / (np.linalg.norm(dir2) + 1e-6)) * mag).tolist()

        for i, p_idx in enumerate(p_to_separate):
            target_v = v1_new if i == 0 else v2_new
            modified_polygons[p_idx] = [target_v if to_t(pt) == v_t else pt for pt in modified_polygons[p_idx]]

        for p_idx in p_to_add_edge:
            new_poly = []
            poly = modified_polygons[p_idx]
            for pt_idx, pt in enumerate(poly):
                if to_t(pt) == v_t:
                    prev_pt = np.array(poly[pt_idx - 1])
                    if np.linalg.norm(np.array(v1_new) - prev_pt) < np.linalg.norm(np.array(v2_new) - prev_pt):
                        new_poly.extend([v1_new, v2_new])
                    else:
                        new_poly.extend([v2_new, v1_new])
                else:
                    new_poly.append(pt)
            modified_polygons[p_idx] = new_poly

    final_output = []
    for poly in modified_polygons:
        pts = np.array([[float(pt[0]), float(pt[1])] for pt in poly])
        if len(pts) >= 3:
            centroid = np.mean(pts, axis=0)
            angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
            sorted_idx = np.argsort(angles)
            final_output.append(pts[sorted_idx].tolist())
        else:
            final_output.append(pts.tolist())

    return final_output


def preprocess_skeleton(mask, k_size=51):
    """Extract skeleton and initial junctions from instance mask."""
    cells_binary = (mask > 0).astype(np.uint8)
    kernel_roi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv2.dilate(cells_binary, kernel_roi)
    filled = ndi.binary_fill_holes(dilated)
    tissue_roi = cv2.erode(filled.astype(np.uint8), kernel_roi)
    tissue_roi = ndi.binary_dilation(tissue_roi, iterations=3)

    cell_boundaries = find_boundaries(mask, mode='inner')
    combined_target = ((mask == 0) & (tissue_roi > 0)) | cell_boundaries
    skeleton = skeletonize(combined_target).astype(np.uint8)

    v_kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton, -1, v_kernel)
    raw_vertices_mask = (neighbor_count >= 13).astype(np.uint8)
    num_v, v_labels, v_stats, v_centroids = cv2.connectedComponentsWithStats(raw_vertices_mask)

    return skeleton, v_labels, v_centroids[1:], raw_vertices_mask


def build_graph(skeleton, v_labels, num_vertices, raw_vertices_mask):
    """Construct a topological graph from skeleton and vertices."""
    skeleton_segments = skeleton.copy()
    skeleton_segments[raw_vertices_mask > 0] = 0
    num_s, s_labels = cv2.connectedComponents(skeleton_segments, connectivity=8)

    G = nx.Graph()
    G.add_nodes_from(range(num_vertices))
    kernel_3x3 = np.ones((3, 3), np.uint8)

    for s_id in range(1, num_s):
        seg_mask = (s_labels == s_id).astype(np.uint8)
        dilated_seg = cv2.dilate(seg_mask, kernel_3x3)
        intersecting_v_ids = np.unique(v_labels[dilated_seg > 0])
        intersecting_v_ids_list = [vid for vid in intersecting_v_ids if vid > 0]
        if len(intersecting_v_ids_list) == 2:
            v1, v2 = int(intersecting_v_ids_list[0] - 1), int(intersecting_v_ids_list[1] - 1)
            G.add_edge(v1, v2)

    while True:
        dead_ends = [n for n, d in G.degree() if d < 2]
        if not dead_ends: break
        G.remove_nodes_from(dead_ends)
    return G


def extract_polygons_room_based(skeleton, v_labels, G, refined_centroids, orig_mask, split_4way=True):
    """Extract polygons from skeleton tiles based on semantic overlap validation."""
    inverted_skeleton = (1 - skeleton).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(inverted_skeleton, connectivity=4)
    v_map_expanded = ndi.maximum_filter(v_labels, size=(3, 3))

    unique_ids, counts = np.unique(orig_mask, return_counts=True)
    instance_areas = dict(zip(unique_ids, counts))

    final_polygons_points = []
    edge_counts: Counter[tuple[int, int]] = Counter()
    temp_cycles = []

    for i in range(1, num_labels):
        room_mask = (labels == i).astype(np.uint8)

        overlap_pixels = orig_mask[room_mask > 0]
        valid_overlap_pixels = overlap_pixels[overlap_pixels > 0]

        if valid_overlap_pixels.size == 0:
            continue

        most_common = Counter(valid_overlap_pixels).most_common(1)
        target_instance_id = most_common[0][0]
        intersection_area = most_common[0][1]

        overlap_ratio = intersection_area / instance_areas[target_instance_id]

        if overlap_ratio <= 0.5:
            continue

        dilated_room = cv2.dilate(room_mask, np.ones((3, 3), np.uint8))
        touched_v_ids = np.unique(v_map_expanded[dilated_room > 0])
        touched_v_ids = [int(v - 1) for v in touched_v_ids if v > 0]
        if len(touched_v_ids) < 3: continue

        sub = G.subgraph(touched_v_ids)
        res_cycles = nx.cycle_basis(sub)
        if res_cycles:
            current_cycle = res_cycles[0]
            temp_cycles.append(current_cycle)
            for j in range(len(current_cycle)):
                u, v = sorted((current_cycle[j], current_cycle[(j + 1) % len(current_cycle)]))
                edge_counts[(u, v)] += 1

    outer_nodes = {node for edge, count in edge_counts.items() if count == 1 for node in edge}
    G_simplified = nx.Graph()
    anchor_nodes = {n for n in G.nodes() if G.degree(n) > 2 or n in outer_nodes}
    G_simplified.add_nodes_from(anchor_nodes)

    for cycle in temp_cycles:
        pts = [refined_centroids[v].tolist() for v in cycle if v in anchor_nodes]
        if len(pts) >= 3:
            final_polygons_points.append(pts)

            cycle_anchors = [v for v in cycle if v in anchor_nodes]
            for j in range(len(cycle_anchors)):
                u_anchor = cycle_anchors[j]
                v_anchor = cycle_anchors[(j + 1) % len(cycle_anchors)]
                G_simplified.add_edge(u_anchor, v_anchor)

    G = G_simplified
    if split_4way:
        final_polygons_points = optimize_four_way_junctions(G, final_polygons_points, refined_centroids)
    return final_polygons_points


class TopologyManager:
    """Manage shared topology points and mesh connectivity."""

    def __init__(self, shapes):
        """Initialize topology points and mesh mapping."""
        self.points = []
        self.point_map = {}
        self.mesh = []

        for shape in shapes:
            shape_point_ids = []
            for pt in shape['points']:
                pt_tuple = (round(float(pt[0]), 4), round(float(pt[1]), 4))
                if pt_tuple not in self.point_map:
                    self.point_map[pt_tuple] = len(self.points)
                    self.points.append(list(pt))
                shape_point_ids.append(self.point_map[pt_tuple])
            self.mesh.append(shape_point_ids)

    def get_shape_points(self, shape_idx):
        """Get coordinates of points for a specific shape index."""
        return np.array([self.points[pid] for pid in self.mesh[shape_idx]])

    def update_point(self, pt_id, new_coords):
        """Update coordinates of a specific point ID."""
        self.points[pt_id] = list(new_coords)


def get_reflex_indices(points):
    """Identify reflex vertices in a polygon using vector cross products."""
    n = len(points)
    if n < 3: return []

    pts = np.asarray(points)
    p_prev = np.roll(pts, 1, axis=0)
    p_curr = pts
    p_next = np.roll(pts, -1, axis=0)

    s = np.sum((p_next[:, 0] - p_curr[:, 0]) * (p_next[:, 1] + p_curr[:, 1]))
    is_cw = s > 0

    v1 = p_curr - p_prev
    v2 = p_next - p_curr
    cp = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    if is_cw:
        return np.where(cp > 1e-6)[0].tolist()
    else:
        return np.where(cp < -1e-6)[0].tolist()


def get_best_projection_vectorized(p, hull):
    """Calculate the shortest projection of a point onto the convex hull vertices."""
    a = hull
    b = np.roll(hull, -1, axis=0)
    ab = b - a
    ap = p - a

    ab_squared = np.sum(ab ** 2, axis=1)
    ab_squared[ab_squared < 1e-8] = 1e-8

    t = np.sum(ap * ab, axis=1) / ab_squared
    t = np.clip(t, 0, 1)

    projections = a + t[:, np.newaxis] * ab
    dists = np.linalg.norm(projections - p, axis=1)

    best_idx = np.argmin(dists)
    return dists[best_idx], projections[best_idx]


def fix_convexity_with_topology(topo):
    """Iteratively fix polygon convexity by moving reflex vertices to convex hull."""
    moved_log = {}
    orig_reflex_map = {}

    for s_idx in range(len(topo.mesh)):
        pts = topo.get_shape_points(s_idx)
        ridx = get_reflex_indices(pts)
        for idx in ridx:
            orig_reflex_map[topo.mesh[s_idx][idx]] = True

    for iteration in range(2):
        for s_idx in range(len(topo.mesh)):
            pts = topo.get_shape_points(s_idx)
            reflex_idx = get_reflex_indices(pts)
            if not reflex_idx: continue

            hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
            for idx in reflex_idx:
                pt_id = topo.mesh[s_idx][idx]
                p_reflex = pts[idx]

                min_dist, best_proj = get_best_projection_vectorized(p_reflex, hull)

                nudge_vec = best_proj - p_reflex
                norm = np.linalg.norm(nudge_vec)
                if norm > 0:
                    best_proj = best_proj + (nudge_vec / norm) * 0.01

                if pt_id not in moved_log or iteration > 0:
                    if pt_id not in moved_log:
                        moved_log[pt_id] = s_idx
                    topo.update_point(pt_id, best_proj)

    return moved_log, orig_reflex_map


def save_json(img_name, points_list, h, w, json_dir):
    """Save extracted polygons to a Labelme-compatible JSON file."""
    labelme_shapes = []
    for idx, points in enumerate(points_list):
        labelme_shapes.append({
            "label": str(idx + 1),
            "points": points,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None
        })
    data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": labelme_shapes,
        "imagePath": img_name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }
    json_path = os.path.join(json_dir, os.path.splitext(img_name)[0] + ".json")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def build_ply_mesh(points_list, image_height):
    """Build a planar PLY mesh using Cartesian coordinates and shared vertex indices."""
    vertices = []
    vertex_map = {}
    faces = []

    for polygon_index, points in enumerate(points_list, start=1):
        face = []
        for point in points:
            x = float(point[0])
            y = float(image_height - 1 - float(point[1]))
            key = (round(x, 4), round(y, 4))

            if key not in vertex_map:
                vertex_map[key] = len(vertices)
                vertices.append((x, y, 0.0))

            vertex_index = vertex_map[key]
            if not face or face[-1] != vertex_index:
                face.append(vertex_index)

        if len(face) > 1 and face[0] == face[-1]:
            face.pop()

        if len(face) < 3 or len(set(face)) < 3:
            raise ValueError(f"Polygon {polygon_index} has fewer than three distinct vertices")
        if len(face) > 255:
            raise ValueError(f"Polygon {polygon_index} has more than 255 vertices")

        area_twice = 0.0
        for i, vertex_index in enumerate(face):
            next_vertex_index = face[(i + 1) % len(face)]
            x1, y1, _ = vertices[vertex_index]
            x2, y2, _ = vertices[next_vertex_index]
            area_twice += x1 * y2 - x2 * y1

        if abs(area_twice) <= 1e-8:
            raise ValueError(f"Polygon {polygon_index} has zero area")
        if area_twice < 0:
            face.reverse()

        faces.append(face)

    return vertices, faces


def save_ply(img_name, points_list, image_height, ply_dir):
    """Save polygons as a standard ASCII PLY vertex-and-face mesh."""
    vertices, faces = build_ply_mesh(points_list, image_height)
    ply_path = os.path.join(ply_dir, os.path.splitext(img_name)[0] + ".ply")

    with open(ply_path, 'w', encoding='ascii', newline='\n') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for x, y, z in vertices:
            f.write(f"{x:.10g} {y:.10g} {z:.10g}\n")
        for face in faces:
            indices = " ".join(str(index) for index in face)
            f.write(f"{len(face)} {indices}\n")


def render_visualization(img_rgb, mask, skeleton, G, refined_centroids, final_polygons, img_name, vis_dir):
    """Render and save pipeline processing status as an image."""
    plt.figure(figsize=(20, 16))

    plt.subplot(2, 2, 1)
    max_m = np.max(mask) if np.max(mask) > 0 else 1
    plt.imshow((plt.get_cmap('nipy_spectral')(mask / max_m)[:, :, :3] * 255).astype(np.uint8))
    plt.title("1. Mask")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    vis_v = cv2.addWeighted(img_rgb, 0.4, np.full(img_rgb.shape, 255, dtype=np.uint8), 0.6, 0)
    for center in refined_centroids:
        c = tuple(center.astype(int))
        cv2.circle(vis_v, c, 11, (255, 255, 255), 2)
        cv2.circle(vis_v, c, 10, (0, 0, 0), -1)
    plt.imshow(vis_v)
    plt.title("2. Junctions")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    vis_deg = cv2.addWeighted(img_rgb, 0.4, np.full(img_rgb.shape, 255, dtype=np.uint8), 0.6, 0)
    vis_deg[skeleton > 0] = [180, 180, 180]
    for u, v in G.edges():
        cv2.line(vis_deg, tuple(refined_centroids[u].astype(int)), tuple(refined_centroids[v].astype(int)),
                 (120, 120, 120), 2)
    for node in G.nodes():
        deg = G.degree(node)
        cv2.circle(vis_deg, tuple(refined_centroids[node].astype(int)), 10, get_node_color(deg), -1)
    plt.imshow(vis_deg)
    plt.title("3. Topology")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    vis_poly = img_rgb.copy()
    cmap = plt.get_cmap('tab20')
    for idx, points in enumerate(final_polygons):
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_poly, [pts], True, [int(c * 255) for c in cmap(idx % 20)[:3]], 3)
    plt.imshow(vis_poly)
    plt.title(f"4. Polygons ({len(final_polygons)})")
    plt.axis('off')

    plt.tight_layout()
    save_path = os.path.join(vis_dir, os.path.splitext(img_name)[0] + "_pipeline.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


def analyze_junctions_from_polygons(img_rgb, polygons, img_name, vis_dir):
    """Detect and visualize junction types (degree 2 and 4) from reconstructed polygons."""
    from collections import defaultdict
    point_to_edges = defaultdict(set)
    for poly in polygons:
        n = len(poly)
        for i in range(n):
            p1 = tuple(np.round(poly[i], 2).tolist())
            p2 = tuple(np.round(poly[(i + 1) % n], 2).tolist())
            edge = frozenset([p1, p2])
            point_to_edges[p1].add(edge)
            point_to_edges[p2].add(edge)

    deg2_points, deg4_points = [], []
    all_unique_edges = set()
    for pt_coords, edges in point_to_edges.items():
        degree = len(edges)
        if degree == 2:
            deg2_points.append(pt_coords)
        elif degree == 4:
            deg4_points.append(pt_coords)
        all_unique_edges.update(edges)

    vis_img = img_rgb.copy()
    for edge in all_unique_edges:
        pts = list(edge)
        if len(pts) == 2:
            p1, p2 = (int(pts[0][0]), int(pts[0][1])), (int(pts[1][0]), int(pts[1][1]))
            cv2.line(vis_img, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)

    for pt in deg2_points:
        center = (int(pt[0]), int(pt[1]))
        cv2.circle(vis_img, center, 6, (255, 255, 0), -1)
    for pt in deg4_points:
        center = (int(pt[0]), int(pt[1]))
        cv2.circle(vis_img, center, 10, (255, 0, 0), -1)

    plt.figure(figsize=(12, 10))
    plt.imshow(vis_img)
    plt.title(f"Yellow: Deg2({len(deg2_points)}), Red: Deg4({len(deg4_points)})")
    plt.axis('off')

    save_path = os.path.join(vis_dir, os.path.splitext(img_name)[0] + "_junctions.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


def batch_process(img_folder, mask_folder, output_dir, vis_out_folder, save_ply_output=True,
                  save_json_output=True, fix_convexity=True, split_4way=True, k_size=51):
    """Process multiple images/masks in a folder sequence."""
    if not save_ply_output and not save_json_output:
        raise ValueError("At least one of PLY or JSON output must be enabled")

    os.makedirs(output_dir, exist_ok=True)
    ply_out_folder = os.path.join(output_dir, "ply") if save_ply_output else None
    json_out_folder = os.path.join(output_dir, "json") if save_json_output else None
    if ply_out_folder:
        os.makedirs(ply_out_folder, exist_ok=True)
    if json_out_folder:
        os.makedirs(json_out_folder, exist_ok=True)
    if vis_out_folder:
        if not img_folder:
            print("Warning: '--vis_dir' is provided but '--img_dir' is missing. Visualization will be disabled.")
            vis_out_folder = None
        else:
            os.makedirs(vis_out_folder, exist_ok=True)

    mask_files = os.listdir(mask_folder)
    img_exts = ('.tif', '.png', '.jpg', '.tiff')

    if img_folder:
        img_files = [f for f in os.listdir(img_folder) if f.lower().endswith(img_exts)]
        print(f"Total images found: {len(img_files)}")
        loop_items = img_files
        is_img_loop = True
    else:
        mask_valid_files = [f for f in mask_files if f.lower().endswith(img_exts)]
        print(f"Total masks found: {len(mask_valid_files)}")
        loop_items = mask_valid_files
        is_img_loop = False

    for item_name in loop_items:
        item_base_name = os.path.splitext(item_name)[0]
        img_path = None
        mask_path = None
        file_identifier = item_name

        if is_img_loop:
            img_path = os.path.join(img_folder, item_name)
            matched_mask = None
            for mf in mask_files:
                if item_base_name in mf:
                    matched_mask = mf
                    break
            if matched_mask is None:
                print(f"Skip: No mask found for {item_name}")
                continue
            mask_path = os.path.join(mask_folder, matched_mask)
            print(f"Processing: {item_name} <-> {matched_mask}")
        else:
            mask_path = os.path.join(mask_folder, item_name)
            print(f"Processing Mask: {item_name}")

        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None: continue

            h, w = mask.shape[:2]
            img_rgb = None

            if img_path:
                img = cv2.imread(img_path)
                if img is None: continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]

            if len(np.unique(mask)) <= 2:
                _, instance_mask = cv2.connectedComponents((mask > 0).astype(np.uint8))
            else:
                instance_mask = mask

            skeleton, v_labels, v_cents, v_mask = preprocess_skeleton(mask, k_size=k_size)
            G = build_graph(skeleton, v_labels, len(v_cents), v_mask)
            polygons = extract_polygons_room_based(skeleton, v_labels, G, v_cents, orig_mask=instance_mask,
                                                   split_4way=split_4way)

            if fix_convexity and len(polygons) > 0:
                print(f"   [Convexity Fix] Applying convex hull fix to {len(polygons)} polygons...")
                shapes_for_topo = [{"points": poly} for poly in polygons]
                topo_manager = TopologyManager(shapes_for_topo)
                fix_convexity_with_topology(topo_manager)

                fixed_polygons = []
                for s_idx in range(len(topo_manager.mesh)):
                    fixed_polygons.append(topo_manager.get_shape_points(s_idx).tolist())
                polygons = fixed_polygons

            if ply_out_folder:
                print(f"   [Save] Saving PLY to {ply_out_folder}...")
                save_ply(file_identifier, polygons, h, ply_out_folder)
            if json_out_folder:
                print(f"   [Save] Saving JSON to {json_out_folder}...")
                save_json(file_identifier, polygons, h, w, json_out_folder)

            if vis_out_folder and img_rgb is not None:
                print(f"   [Render] Rendering pipeline visualization...")
                render_visualization(img_rgb, mask, skeleton, G, v_cents, polygons, file_identifier, vis_out_folder)

                print(f"   [Render] Analyzing junctions...\n")
                analyze_junctions_from_polygons(img_rgb, polygons, file_identifier, vis_out_folder)
            else:
                print()

        except Exception as e:
            import traceback
            print(f"Error processing {item_name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCPCN Polygon Topology Processing Pipeline")
    parser.add_argument("-i", "--img_dir", type=str, default=None,
                        help="(Optional) Input directory of original images, required for visualization")
    parser.add_argument("-m", "--mask_dir", type=str, required=True, help="Input directory of instance masks")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Output root directory; PLY and JSON files are saved in separate subdirectories")
    parser.add_argument("-v", "--vis_dir", type=str, default=None,
                        help="(Optional) Output directory for visualization images")
    parser.add_argument("--save_ply", type=int, choices=[0, 1], default=1,
                        help="Generate PLY files (1=Enable, 0=Disable, default: 1)")
    parser.add_argument("--save_json", type=int, choices=[0, 1], default=1,
                        help="Generate LabelMe JSON files (1=Enable, 0=Disable, default: 1)")

    parser.add_argument("--fix_convexity", type=int, choices=[0, 1], default=1,
                        help="Enable automatic convex hull fix (1=Enable, 0=Disable, default: 1)")
    parser.add_argument("--split_4way", type=int, choices=[0, 1], default=1,
                        help="Enable 4-way junction optimization (1=Enable, 0=Disable, default: 1)")
    parser.add_argument("--k_size", type=int, default=51,
                        help="Kernel size for dilation in skeleton preprocessing (default: 51)")

    args = parser.parse_args()
    if not args.save_ply and not args.save_json:
        parser.error("--save_ply and --save_json cannot both be 0")

    print("\n--- Current Pipeline Configuration ---")
    print(f"Image Dir : {args.img_dir}")
    print(f"Mask Dir  : {args.mask_dir}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Save PLY  : {bool(args.save_ply)}")
    print(f"Save JSON : {bool(args.save_json)}")
    print(f"Vis Dir   : {args.vis_dir if args.vis_dir else 'Disabled'}")
    print(f"Fix Convex: {bool(args.fix_convexity)}")
    print(f"Split 4Way: {bool(args.split_4way)}")
    print(f"K-Size    : {args.k_size}")
    print("--------------------------------------\n")

    batch_process(
        args.img_dir,
        args.mask_dir,
        args.output_dir,
        args.vis_dir,
        save_ply_output=bool(args.save_ply),
        save_json_output=bool(args.save_json),
        fix_convexity=bool(args.fix_convexity),
        split_4way=bool(args.split_4way),
        k_size=args.k_size
    )
    print("All tasks finished successfully.")
