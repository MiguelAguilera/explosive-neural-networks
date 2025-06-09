# Explosive Neural Networks

**Explosive Neural Networks via Higher‑Order Interactions in Curved Statistical Manifolds**

This repository accompanies the paper by Aguilera et al. (2025), presenting a novel class of associative memory networks with **explosive memory-retrieval transitions**, enabled by **higher-order interactions** and **curved statistical geometry**.

## Overview

This codebase supports the generation and visualization of results from the paper. The models:
- Generalize Hopfield-like neural networks using a deformed maximum entropy principle.
- Exhibit explosive transitions, multi-stability, and memory capacity enhancements.
- Are analytically tractable via mean-field and replica methods.

## Repository Structure

```
.
├── data/                         # Precomputed data for figures
├── img/                          # Output image files
├── Fig2a.py                      # Exokisuve phase transitions
├── Fig2b.py
├── Fig2c.py
├── Fig2d.py
├── Fig3a.py                      # Bistable attractors and energy landscapes
├── Fig3b.py
├── Fig4.py                       # Replica-based memory capacity phase diagram
├── Fig5.py                       # CIFAR image encoding and memory retrieval stats
├── Fig6.py                       # Explosive spin-glass transitions
├── generate_data_Fig2c.py
├── generate_data_Fig3b.py
├── generate_data_Fig4.py
├── generate_data_Fig5.py
├── generate_data_Fig6.py
├── generate_dataset_Fig5.py
├── LICENSE
└── README.md
```

## Usage

Run simulations and plotting scripts for each figure:

```bash
python Fig4.py
```

To regenerate the associated data:

```bash
python generate_data_Fig4.py
```

Images will be saved in the `img/` directory, and datasets in `data/`.

## Figures Summary

- **Figure 2 (a–d)**: Phase transitions and emergence of explosive behavior from mean-field analysis.
- **Figure 3 (a–b)**: Memory attractor dynamics and hysteresis with two correlated patterns.
- **Figure 4**: Replica theory phase diagram showing extended memory capacity in deformed models.
- **Figure 5**: Empirical validation using CIFAR-100 patterns—retrieval stability under different deformations.
- **Figure 6**: Spin-glass model with explosive transitions driven by higher-order interaction geometry.

## Citation

Please cite the corresponding preprint:

> Aguilera, M., Morales, P.A., Rosas, F.E., Shimazaki, H. (2025). Explosive neural networks via higher-order interactions in curved statistical manifolds. _Nature Communications_.

## License

Distributed under the GNU General Public License v3.0.
