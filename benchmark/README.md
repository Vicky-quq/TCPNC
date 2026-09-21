# Benchmark Data and Results

This directory documents the datasets and generated results used to evaluate TCPCN. Large files are distributed separately so that the source repository remains lightweight.

## Downloads

- **Evaluation datasets:** hosted on Zenodo at [https://zenodo.org/records/20301518](https://zenodo.org/records/20301518) (DOI: [10.5281/zenodo.20301518](https://doi.org/10.5281/zenodo.20301518)).
- **Generated results:** available from the [TCPNC GitHub Releases](https://github.com/Vicky-quq/TCPNC/releases) page.

After downloading, extract the dataset archive into `benchmark/Dataset/` and the result archives into `benchmark/`. The reconstructed local layout should match the structure below.

## Directory Structure

```text
benchmark/
├── README.md
├── Dataset/
│   ├── Sansha-5/
│   │   ├── *.tif
│   │   └── *.json
│   └── WO115-2/
│       ├── *.tiff
│       └── *.json
└── Result/
    ├── Sansha-5/
    │   ├── cellposesam/
    │   ├── cellposesam_ours/
    │   │   ├── json/
    │   │   └── ply/
    │   ├── cyto3/
    │   └── cyto3_ours/
    │       ├── json/
    │       └── ply/
    ├── WO115-2/
    │   └── ...
    ├── *.npy
    ├── *.csv
    ├── *.txt
    └── *.pdf
```

The JSON files under `Dataset/` are LabelMe ground-truth annotations. The `cellposesam/` and `cyto3/` directories contain instance segmentation masks. Directories ending in `_ours` contain TCPCN post-processing outputs: LabelMe polygons and ASCII PLY meshes.

## Release Assets

The generated results are split by dataset and segmentation method to simplify downloading:

- `Sansha-5_cellposesam.zip` and `Sansha-5_cyto3.zip` contain the original instance masks for Sansha-5.
- `Sansha-5_cellposesam_tcpcn_geometry.zip` and `Sansha-5_cyto3_tcpcn_geometry.zip` contain the corresponding TCPCN JSON and PLY outputs for Sansha-5.
- `WO115-2_cellposesam.zip` and `WO115-2_cyto3.zip` contain the original instance masks for WO115-2.
- `WO115-2_cellposesam_tcpcn_geometry.zip` and `WO115-2_cyto3_tcpcn_geometry.zip` contain the corresponding TCPCN JSON and PLY outputs for WO115-2.
- `evaluation_summaries.zip` contains the CSV, TXT, and PDF evaluation summaries under `Result/`.
- Each `*.npy.zip` archive contains one segmentation-evaluation array and extracts directly under `Result/`.
- `SHA256SUMS.txt` records checksums for all release archives.

Extract every archive into the `benchmark/` directory. Existing directory names and filenames should be preserved because the evaluation commands use them to associate datasets and methods. Full pipeline visualizations are not distributed as Release assets.

## Reproducing TCPCN Outputs

The following example applies TCPCN to Cellpose-SAM masks from Sansha-5:

```bash
python TCPCN.py \
  --img_dir benchmark/Dataset/Sansha-5 \
  --mask_dir benchmark/Result/Sansha-5/cellposesam \
  --output_dir benchmark/Result/Sansha-5/cellposesam_ours \
  --fix_convexity 1 \
  --split_4way 1 \
  --k_size 51
```

## Reproducing Evaluation Outputs

Generate the segmentation arrays for one dataset:

```bash
python -m evaluation.save_to_npy \
  --pred_root benchmark/Result/Sansha-5 \
  --gt_root benchmark/Dataset/Sansha-5 \
  --output_dir benchmark/Result \
  --overlap_threshold 0.3
```

Generate the segmentation summary figure from the downloaded NPY files:

```bash
python -m evaluation.segmentation.violin \
  --result_dir benchmark/Result \
  --output_dir benchmark/Result \
  --datasets Sansha-5 WO115-2 \
  --algs cyto3 cyto3_ours cellposesam cellposesam_ours
```

Run the boundary evaluation:

```bash
python -m evaluation.boundary.BF1S \
  --gt_root benchmark/Dataset \
  --pred_root benchmark/Result \
  --output_dir benchmark/Result \
  --algs cyto3 cyto3_ours cellposesam cellposesam_ours
```

Run commands from the repository root. The datasets and results are not required to use TCPCN on other data; they are provided only for reproducing the reported evaluation.
