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
    │   │   ├── ply/
    │   │   └── vis/
    │   ├── cyto3/
    │   └── cyto3_ours/
    │       ├── json/
    │       ├── ply/
    │       └── vis/
    ├── WO115-2/
    │   └── ...
    ├── *.npy
    ├── *.csv
    ├── *.txt
    └── *.pdf
```

The JSON files under `Dataset/` are LabelMe ground-truth annotations. The `cellposesam/` and `cyto3/` directories contain instance segmentation masks. Directories ending in `_ours` contain TCPCN post-processing outputs: LabelMe polygons, ASCII PLY meshes, and pipeline visualizations.

## Release Assets

The generated results are split into multiple ZIP archives to simplify downloading:

- `Sansha-5_results.zip` extracts to `Result/Sansha-5/`.
- `WO115-2_results.zip` extracts to `Result/WO115-2/`.
- `evaluation_summaries.zip` contains the CSV, TXT, and PDF evaluation summaries under `Result/`.
- Each `*.npy.zip` archive contains one segmentation-evaluation array and extracts directly under `Result/`.
- `SHA256SUMS.txt` records checksums for all release archives.

Extract every archive into the `benchmark/` directory. Existing directory names and filenames should be preserved because the evaluation commands use them to associate datasets and methods.

## Reproducing TCPCN Outputs

The following example applies TCPCN to Cellpose-SAM masks from Sansha-5:

```bash
python TCPCN.py \
  --img_dir benchmark/Dataset/Sansha-5 \
  --mask_dir benchmark/Result/Sansha-5/cellposesam \
  --output_dir benchmark/Result/Sansha-5/cellposesam_ours \
  --vis_dir benchmark/Result/Sansha-5/cellposesam_ours/vis \
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
