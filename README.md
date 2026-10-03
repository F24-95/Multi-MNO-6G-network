# AI-Based Dynamic Spectrum Sharing for Multi-MNO 6G Networks

## Overview
A software simulation prototype of an AI-driven dynamic spectrum sharing system across 4 Mobile Network Operators (MNOs) over 16 channels in a simulated 6G environment.

## Architecture
- **Spectrum Sensing:** Unsupervised VAE-GMM pipeline
- **Feature Extraction:** Denoising Bottleneck Autoencoder
- **Synthetic Data Augmentation:** Vanilla GAN on minority demand/shortage windows
- **Traffic Forecasting:** Dual LSTM / GRU networks
- **AI Decision Engine:** Deterministic constrained allocator enforcing fairness and stability constraints
- **Backend & Dashboard:** Flask API + PostgreSQL + Chart.js dashboard
