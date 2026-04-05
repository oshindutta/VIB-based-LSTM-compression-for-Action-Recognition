# VIB-LSTM: Compressing Sequential Networks for Action Recognition

<p align="center">
  <a href="https://openaccess.thecvf.com/content/WACV2021/html/Srivastava_A_Variational_Information_Bottleneck_Based_Method_to_Compress_Sequential_Networks_WACV_2021_paper.html"><img src="https://img.shields.io/badge/WACV%202021-Published-blue" alt="WACV 2021"/></a>
  <a href="https://arxiv.org/abs/2010.01343"><img src="https://img.shields.io/badge/arXiv-2010.01343-b31b1b" alt="arXiv"/></a>
  <img src="https://img.shields.io/badge/PyTorch-1.2%2B-orange" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3.6%2B-blue" alt="Python"/>
</p>

Official implementation of **[A Variational Information Bottleneck Based Method to Compress Sequential Networks for Human Action Recognition](https://openaccess.thecvf.com/content/WACV2021/html/Srivastava_A_Variational_Information_Bottleneck_Based_Method_to_Compress_Sequential_Networks_WACV_2021_paper.html)**, published at **IEEE/CVF WACV 2021**. [[arXiv]](https://arxiv.org/abs/2010.01343)

> **TL;DR:** VIB layers placed inside each LSTM gate learn which hidden dimensions are redundant, compressing sequential networks by up to **16,360×** while maintaining accuracy on standard action recognition benchmarks.

---

## Table of Contents

- [Introduction](#introduction)
- [Key Results](#key-results)
- [Edge Deployment](#edge-deployment-inference-on-raspberry-pi-3)
- [Installation](#installation)
- [Usage](#usage)
  - [CNN-LSTM Compression](#compress-a-cnn-lstm-model)
  - [End-to-End LSTM Compression](#compress-an-end-to-end-lstm-model)
  - [Evaluation](#evaluate-a-trained-model)
  - [Model Conversion](#convert-vib-sparse-model-to-dense-compressed)
- [Notes](#notes)
- [Citation](#citation)

---

## Introduction

Action recognition from video requires both spatial and temporal processing, making CNN-LSTM models large and resource-intensive — often unsuitable for edge deployment. Existing LSTM compression methods (TT-LSTM, BT-LSTM, TR-LSTM) reduce only the input-to-hidden matrix, leaving hidden state redundancy unaddressed. Group-lasso methods target hidden states but ignore the large input dimension. **No prior work compresses both simultaneously.**

This paper proposes a **Variational Information Bottleneck (VIB)**-based pruning approach that places learnable bottleneck variables directly inside each LSTM gate. The model jointly prunes the input feature vector and the hidden state vector in a single end-to-end training pass. VIB layers are removed entirely at inference — only the compact LSTM matrices remain.

### Contributions

1. **Novel VIB-LSTM structure** — trains high-accuracy sparse LSTM models by applying information bottlenecks at every gate (input, forget, cell, output).
2. **Principled sequential compression pipeline** — sparsifies pre-trained RNN/LSTM/GRU weight matrices with minimal hyperparameter tuning.
3. **CNN-LSTM compression framework** — extends VIB to compress the feature input dimension of CNN-LSTM architectures, not just hidden states.
4. **State-of-the-art compression with comparable accuracy** — validated on UCF11, HMDB51, and UCF101; outperforms all prior LSTM compression methods by a large margin.

<table align="center">
  <tr>
    <td align="center" width="60%">
      <img src="assets/architecture_e2e.png" alt="VIB-LSTM cell" width="420"/>
      <br/><em>VIB-LSTM cell: z<sub>i</sub>, z<sub>f</sub>, z<sub>g</sub>, z<sub>o</sub> prune irrelevant input features (bottom) and hidden state dimensions (right) during training.</em>
    </td>
    <td align="center" width="40%">
      <img src="assets/matrices_comp.png" alt="Sparse to compact LSTM weight matrices" width="260"/>
      <br/><em>After pruning (UCF11): W<sub>ih</sub> 57,600 → 1,312 and W<sub>hh</sub> 256 → 33.</em>
    </td>
  </tr>
</table>

---

## Key Results

### End-to-End LSTM — UCF11

| Method | LSTM Parameters | Compression Ratio | Accuracy |
|---|---|---|---|
| Naïve-LSTM | 59.25M | 1× | 69.7% |
| TT-LSTM | 0.268M | 221× | 79.6% |
| BT-LSTM | 0.268M | 221× | 85.3% |
| TR-LSTM | 0.267M | 221.8× | 86.9% |
| HT-LSTM | 0.266M | 222.7× | 87.9% |
| **VIB-LSTM (Ours)** | **0.178M** | **332×** | **85.4%** |

### CNN-LSTM — UCF11

| Method | Total Params | LSTM Params | Accuracy |
|---|---|---|---|
| Naïve-LSTM | 55.38M | 33.57M | 98.53% |
| TR-LSTM | 23.15M | 1.340M | 93.8% |
| ISS | 21.97M | 0.160M | 94.6% |
| **VIB-LSTM (Ours)** | **21.86M** | **2,052** | **98.53%** |
| VIB-LSTM + ISS | 21.86M | 1,680 | 90.2% |
| EfficientNet + VIB-LSTM | 7.8M | 1,680 | 96.6% |

> VIB-LSTM achieves **16,360× compression** of LSTM parameters over Naïve-CNN-LSTM while maintaining full accuracy — outperforming TR-LSTM (25×) and ISS (210×) by orders of magnitude.

<p align="center">
  <img src="assets/compression_ratio_comparison.png" alt="Compression ratio comparison bar chart" width="520"/>
</p>

### HMDB51 and UCF101

| Method | HMDB51 LSTM Params | HMDB51 Acc | UCF101 LSTM Params | UCF101 Acc |
|---|---|---|---|---|
| Naïve-LSTM | 33.57M | 62.9% | 9.44M | 92.6% |
| TR-LSTM | 1.340M | 63.8% | — | — |
| Attention-ConvLSTM | 4.72M | 67.1% | 4.72M | 92.88% |
| **VIB-LSTM (Ours)** | **0.41M** | **64.5%** | **0.46M** | **92.2%** |
| **TS-VIB-LSTM (Ours)** | **0.41M** | **68.16%** | **0.46M** | **93.15%** |

> VIB-LSTM compresses Naïve-LSTM by **81× on HMDB51** and **20× on UCF101**, with accuracy matching or exceeding the uncompressed baseline.

---

## Edge Deployment: Inference on Raspberry Pi 3

The compressed model was benchmarked on a **Raspberry Pi 3 (no parallelism)** on a single UCF11 video, demonstrating real-world edge viability.

| | Uncompressed CNN-LSTM | Compressed CNN-VIB-LSTM |
|---|---|---|
| **LSTM Parameters** | 33.57M | 2,052 |
| **Top-1 Accuracy** | 98.53% | **98.53%** |
| **Inference Time** | 1.26 s | **13 ms** |
| **Speedup** | — | **~100×** |

<table align="center">
  <tr>
    <td align="center">
      <img src="assets/demo_cnn_lstm.gif" alt="Uncompressed CNN-LSTM demo" width="380"/>
      <br/><em>Uncompressed CNN-LSTM</em>
    </td>
    <td align="center">
      <img src="assets/demo_comp_cnn_lstm.gif" alt="Compressed CNN-VIB-LSTM demo" width="380"/>
      <br/><em>Compressed CNN-VIB-LSTM</em>
    </td>
  </tr>
</table>

---

## Installation

**Requirements:** Python 3.6+, PyTorch 1.2+

```bash
pip install -r requirements.txt
```

**Optional — EfficientNet feature extractor:**

```bash
pip install efficientnet_pytorch
# or clone: https://github.com/lukemelas/EfficientNet-PyTorch
```

---

## Usage

All scripts accept positional arguments. Paths with defaults can be overridden; required arguments will error if omitted. Run any script with `bash scripts/<name>.sh --help` to see its inline usage comment.

### Compress a CNN-LSTM Model

**Train and prune from scratch:**
```bash
bash scripts/train_cnn_vib_scratch.sh <dataset_path> <save_dir>
```

**Prune a pre-trained CNN-LSTM model:**
```bash
bash scripts/train_cnn_vib_pretrained.sh <dataset_path> <path_to_model> <save_dir>
```

**Fine-tune a pruned CNN-VIB-LSTM model:**
```bash
bash scripts/finetune_cnn_vib.sh <dataset_path> <path_to_model> <save_dir>
```

**Resume pruning from a checkpoint:**
```bash
bash scripts/resume_cnn_vib.sh <dataset_path> <path_to_checkpoint> <save_dir>
```

---

### Compress an End-to-End LSTM Model

**Train and prune from scratch:**
```bash
bash scripts/train_e2e_vib_scratch.sh <dataset_path> <save_dir>
```

**Prune a pre-trained end-to-end LSTM:**
```bash
bash scripts/train_e2e_vib_pretrained.sh <dataset_path> <path_to_model> <save_dir>
```

**Fine-tune a pruned end-to-end VIB-LSTM:**
```bash
bash scripts/finetune_e2e_vib.sh <dataset_path> <path_to_model> <save_dir>
```

**Resume pruning from a checkpoint:**
```bash
bash scripts/resume_e2e_vib.sh <dataset_path> <path_to_checkpoint> <save_dir>
```

---

### Evaluate a Trained Model

```bash
bash scripts/evaluate.sh <dataset_path> <path_to_checkpoint> <save_dir>
```

> Adjust `--img_dim1` / `--img_dim2` inside [scripts/evaluate.sh](scripts/evaluate.sh) to match the model's expected input size.

---

### Convert VIB Sparse Model to Dense-Compressed

```bash
bash scripts/convert_to_dense.sh <path_to_VIB_model> <dataset_path>
```

---

## Notes

- **Dataset loader:** `datasetucf11.py` may need adjustments for your dataset folder structure. `--dataset_path` accepts the path to the dataset root (e.g., `UCF11_updated_mg-frames` for UCF11).
- **Compression factor:** `--kml` sets the compression factor multiplier. It is kept flexible per layer and can be modified in `model_VIBLSTM.py`.
- **Selective pruning:** VIB pruning can be done in steps:
  - To prune only the LSTM input / feature dimension: set `masking=True` in `ib0`, `ib1`, `ib2`, `ib3` inside `model_VIBLSTM.py`.
  - To prune only the hidden state dimension: set `masking=True` in `ib4` inside `model_VIBLSTM.py`.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{srivastava2021variational,
  title={A variational information bottleneck based method to compress sequential networks for human action recognition},
  author={Srivastava, Ayush and Dutta, Oshin and Gupta, Jigyasa and Agarwal, Sumeet and AP, Prathosh},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={2745--2754},
  year={2021}
}
```

---

## Author & Contact

**Oshin Dutta**
- LinkedIn: [https://www.linkedin.com/in/oshindutta/](https://www.linkedin.com/in/oshindutta/)
- Email: [oshind.phd@gmail.com](mailto:oshind.phd@gmail.com)
