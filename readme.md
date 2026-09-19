# TCPNC

TCPNC is a polygon topology reconstruction and evaluation toolkit for cell instance masks. The main pipeline is `TCPCN.py` and exports polygon meshes as PLY and LabelMe JSON files; scripts under `evaluation/` are evaluation utilities for segmentation, boundary, and topology analysis.

## Directory Layout

```text
TCPNC/
├── TCPCN.py                         # Main polygon topology reconstruction pipeline
└── evaluation/
    ├── tools.py                     # Shared evaluation helpers
    ├── save_to_npy.py               # Generate segmentation metric NPY files
    ├── boundary/
    │   ├── BF1S.py                  # Boundary precision/recall/F-score evaluation
    │   └── mean_diameter.py         # Estimate mean/median GT cell diameter
    ├── segmentation/
    │   └── violin.py                # Draw segmentation AP/error violin plots
    └── topology/
        ├── NFCS.py                  # Neighborhood feature consistency score
        └── polygon_analysis.py      # Polygon vertex-count distribution analysis
```

`evaluation/tools.py` is the shared dependency for evaluation scripts. Do not use `eval_utils.py`; it has been removed.

## Environment

Python 3.9+ is recommended. Create an isolated environment first:

```bash
conda create -n tcpcn python=3.10
conda activate tcpcn
```

Install the common dependencies:

```bash
pip install numpy opencv-python matplotlib networkx scikit-image scipy shapely pandas tifffile labelme
```

The evaluation scripts also import `cellpose4.metrics` and `cellpose4.utils`. The required Cellpose4 package can be downloaded from the [MouseLand/cellpose](https://github.com/MouseLand/cellpose) repository. For example, install it from source with:

```bash
git clone https://github.com/MouseLand/cellpose.git
cd cellpose
pip install -e .
```

After installation, make sure this check succeeds:

```bash
python -c "import cellpose4"
```

You can verify the rest of the environment with:

```bash
python -c "import cv2, numpy, matplotlib, networkx, skimage, scipy, shapely, pandas, tifffile, labelme"
```

## Data Organization

Recommended input layout:

```text
data/
├── images/
├── masks/
├── gt/
│   ├── Sansha-5/*.json
│   └── WO115-2/*.json
└── predictions/
    ├── Sansha-5/
    │   ├── cyto3/*.json
    │   ├── cyto3_ours/*.json
    │   ├── cellposesam/*.json
    │   └── cellposesam_ours/*.json
    └── WO115-2/
        └── ...
```

Ground-truth JSON files should follow the LabelMe format and include `imageHeight`, `imageWidth`, and polygon-like shapes.

### Image–Mask Pairing Requirements

When `--img_dir` is provided, TCPCN processes every supported image file in that directory and searches for a mask file in `--mask_dir` whose filename contains the image basename. For example, `A2 0808.tif` can be paired with `A2 0808_masks.tif`.

To ensure reliable pairing:

- Store original images and instance masks in **separate directories**. Do not point both `--img_dir` and `--mask_dir` to the same directory; otherwise mask files may also be processed as images.
- Use a consistent, unique naming convention, such as `sample.tif` and `sample_masks.tif`.
- Ensure that each image basename matches **exactly one** mask filename. Avoid multiple candidates such as `sample_masks.tif` and `sample_masks_old.tif`, because the current pipeline uses the first matching mask file it finds.
- Keep the image and its corresponding mask at the same pixel dimensions. Masks should be single-channel instance-label images: background `0`, with each cell assigned a distinct positive integer label.

## Practical Scope and Limitations

TCPCN is designed for the topological post-processing of static two-dimensional (2D) cell instance segmentation masks. It generally provides more stable reconstructions when individual cell contours within a tissue can be reasonably approximated by low-complexity polygons with relatively few sides and exhibit limited local concavity. For cells with highly curved contours, deep local concavities, or highly irregular shapes, automated reconstruction performance may be limited, and users may need to inspect and manually correct the results. For low-quality masks containing substantial noise, incomplete boundaries, or segmentation errors, improving the segmentation or using complementary tools is recommended. In addition, TCPCN does not perform cross-frame cell tracking or temporal-consistency analysis; dynamic datasets therefore require appropriate temporal analysis tools.

## Run TCPCN

Convert instance masks to polygon meshes. PLY and LabelMe JSON outputs are both enabled by default:

```bash
python TCPCN.py \
  --mask_dir data/masks \
  --output_dir results \
  --img_dir data/images \
  --vis_dir results/vis \
  --save_ply 1 \
  --save_json 1 \
  --fix_convexity 1 \
  --split_4way 1 \
  --k_size 51
```

`--img_dir` and `--vis_dir` are optional. Provide both to save the standard pipeline and junction visualizations; otherwise omit them.
Generated files are placed in `results/ply/` and `results/json/`. Use `--save_ply 0` or `--save_json 0` to disable either format. The two options cannot both be `0`.
When standard visualization is enabled, TCPCN saves `<filename>_pipeline.png` and `<filename>_junctions.png` in `--vis_dir`.

PLY files use the standard ASCII PLY vertex-and-face structure:

```text
element vertex N
property float x
property float y
property float z
element face M
property list uchar int vertex_indices
```

PLY coordinates use a Cartesian coordinate system with the origin at the lower left, positive Y upward, and `z = 0`. Faces use counterclockwise vertex order. Shared polygon vertices are stored once and referenced by index from each face. JSON files retain the original image coordinate system with the origin at the upper left and positive Y downward.

### Optional Topology-Refinement Switches

Both topology-refinement steps can be enabled or disabled independently:

- `--fix_convexity 1|0`: apply convexity correction to reconstructed polygons (`1` by default).
- `--split_4way 1|0`: split four-way junctions into three-way junctions during polygon reconstruction (`1` by default).

Set either option to `0` to disable the corresponding step.

## Segmentation Evaluation

Generate `.npy` metric files for one dataset. `--pred_root` should contain one subfolder per method:

```bash
python -m evaluation.save_to_npy \
  --pred_root data/predictions/Sansha-5 \
  --gt_root data/gt/Sansha-5 \
  --output_dir results/segmentation \
  --overlap_threshold 0.3
```

Draw AP/error violin plots from generated `.npy` files:

```bash
python -m evaluation.segmentation.violin \
  --result_dir results/segmentation \
  --output_dir results/figures \
  --datasets Sansha-5 WO115-2 \
  --algs cyto3 cyto3_ours cellposesam cellposesam_ours
```

## Boundary Evaluation

Compute mean and median GT cell diameters:

```bash
python -m evaluation.boundary.mean_diameter \
  --gt_root data/gt \
  --datasets Sansha-5 WO115-2
```

Run boundary precision, recall, and F-score evaluation:

```bash
python -m evaluation.boundary.BF1S \
  --gt_root data/gt \
  --pred_root data/predictions \
  --output_dir results/boundary \
  --algs cyto3 cyto3_ours cellposesam cellposesam_ours
```

## Topology Evaluation

Evaluate NFCS for a single dataset and method:

```bash
python -m evaluation.topology.NFCS \
  --gt_dir data/gt/Sansha-5 \
  --pred_dir data/predictions/Sansha-5/cyto3_ours
```

Analyze polygon vertex-count distributions and export CSV files:

```bash
python -m evaluation.topology.polygon_analysis \
  --pred_root data/predictions \
  --gt_root data/gt \
  --datasets Sansha-5 WO115-2 \
  --algs cellposesam_ours cyto3_ours \
  --roi_threshold 0.5
```

Run commands from the `TCPNC/` directory so that the `evaluation` package imports resolve correctly.
