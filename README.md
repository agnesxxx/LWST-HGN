# LWST-HGN

Official-style release code for the **UCI Air Quality C6H6(GT) soft-sensing task**.
This repository contains the cleaned single-task version corresponding to the
best reported AirQuality result.

## Repository structure

```text
LWST-HGN-AirQuality/
├── train_airquality_c6h6.py      # main training/evaluation code
├── run_best.sh                   # exact command for the released best setting
├── configs/
│   └── best_c6h6.json            # human-readable best configuration
├── results/
│   └── reference_metrics.json    # reference best metrics
├── data/
│   └── README.md                 # dataset placement instructions
├── requirements.txt
├── LICENSE
└── README.md
```

## Environment

The original experiments were run with PyTorch 1.11.0 + CUDA 11.3. A compatible
PyTorch installation is required. The remaining Python dependencies can be
installed with:

```bash
pip install -r requirements.txt
```

## Dataset

Download the **UCI Air Quality** dataset and place the CSV at:

```text
data/AirQualityUCI.csv
```

The code uses the strict input protocol:

```text
PT08_S1_CO, PT08_S2_NMHC, PT08_S3_NOx, PT08_S4_NO2,
PT08_S5_O3, T, RH, AH
```

Correlation screening is performed **using the training split only**. In the
reference run, the three removed variables are `AH`, `RH`, and `T`.

## Reproduce the best result

The default arguments in `train_airquality_c6h6.py` already correspond to the
released best configuration. Therefore, the shortest command is:

```bash
python -u train_airquality_c6h6.py \
  --data_path data/AirQualityUCI.csv \
  --gpu 0
```

Alternatively, use the explicit reproduction script:

```bash
bash run_best.sh 0 data/AirQualityUCI.csv
```

Change the first argument to another GPU index when needed.

## Key configuration

| Component | Setting |
|---|---|
| Target | `C6H6_GT` |
| Split | chronological 60/20/20 |
| Window length | 3 |
| Pearson screening | remove 3 lowest-correlation inputs |
| Hypergraph | static, `k=3` |
| MS-GTC kernels | `{3,5,7}` |
| MS-GTC fusion | gated |
| Wavelet initialization | db2 |
| Residual enhancement scale | `alpha=0.055` |
| Low-frequency ratio | `rho=0.5` |
| Orthogonality weight | `1e-5` |
| Energy weight | `1e-4` |
| Loss | Huber |
| Learning rate | `1.5e-3` |
| Weight decay | `1e-4` |
| Batch size | 128 |
| Dropout | 0.1 |
| Seed | 3407 |

## Reference result

The released best run achieved:

| Split | NMAE | NRMSE | MAPE | R2 |
|---|---:|---:|---:|---:|
| Validation | 0.427638 | 0.926696 | 4.363315 | 0.995141 |
| Test | **0.768057** | **1.074252** | 7.656405 | **0.995024** |

The selected checkpoint occurred at epoch 147 in the reference run. Minor
numerical variation can occur across CUDA, cuDNN, GPU, and PyTorch versions.

## Outputs

Each run creates a timestamped folder under `outputs/`. The task folder stores:

- `summary.json`: configuration, selected `k`, metrics, and learned-wavelet diagnostics;
- `history_best_k.json`: epoch-wise training/validation history;
- `predictions.npz` / `predictions.csv`: validation and test predictions;
- optional adjacency/model files when enabled through CLI arguments.

## Notes on reproducibility

- The train/validation/test split is chronological; no random split is used.
- Pearson screening uses only the training set to avoid information leakage.
- Input normalization and target normalization use training statistics only.
- `C6H6_GT` is transformed by `log1p` before standardization and is restored to
  the original scale before metric computation.
- The high-pass wavelet filter is not independently learned; it is generated
  from the learnable low-pass prototype through the QMF relation.

## License

MIT License. Replace the copyright line with the final author/team information
before public release if desired.
