# TCPNC

TCPNC is a polygon topology reconstruction and evaluation toolkit for cell instance masks. The main pipeline is `TCPCN.py`; scripts under `evaluation/` are evaluation utilities for segmentation, boundary, and topology analysis.

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

## Run TCPCN

Convert instance masks to topology-aware polygon JSON files:

```bash
python TCPCN.py \
  --mask_dir data/masks \
  --json_dir results/json \
  --img_dir data/images \
  --vis_dir results/vis \
  --fix_convexity 1 \
  --split_4way 1 \
  --k_size 51
```

`--img_dir` and `--vis_dir` are optional. If visualization is not needed, omit both.

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
