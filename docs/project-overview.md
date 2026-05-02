# NOCM Project Overview

## The Problem We're Trying to Solve

Diffusion models generate images by starting with pure noise and gradually removing it over many steps. Mathematically, there's a forward process that adds noise to a clean image — `x_t = α_t x_0 + σ_t ε` where `ε` is Gaussian noise and `t` goes from 0 (clean) to T (pure noise) — and then a reverse process that undoes this, governed by an ODE whose solution traces a trajectory from noise back to a clean image. The neural network learns to predict the "velocity" or "score" at each point along this trajectory, and you integrate the ODE step by step. The catch is that this requires hundreds of network evaluations to get a good image, which is slow and expensive.

A whole line of work (consistency models, flow matching, distillation methods) tries to compress this into 1–4 steps. Consistency models do this by enforcing that the network's prediction is the same regardless of where along the trajectory you query it — if `x_t` and `x_r` are two points on the same denoising path, then `f(x_t, t) = f(x_r, r) = x_0`. This "consistency property" means at inference time you can just evaluate once from any noise level and get the clean image directly.

## Where Neural Operators Come In

Separately, there's a field called neural operators — architectures (particularly Fourier Neural Operators / FNOs) designed to learn mappings between functions, not just between fixed-size vectors. They work by projecting inputs into the frequency domain via FFT, applying learned spectral convolutions, and projecting back. Their key property is **discretization invariance**: train at one resolution, evaluate at another. They were originally developed for solving PDEs (Navier-Stokes, Burgers, etc.), and the core insight of our project is that the diffusion reverse process IS an ODE — so a neural operator should, in principle, be able to solve it.

## The NOCM Project Thesis

NOCM (Neural Operator Consistency Models) combines these two ideas: use a neural operator architecture to learn a consistency model for image generation. The potential payoffs are significant — you'd get few-step generation (from the consistency property) with resolution invariance (from the neural operator), which no existing method provides.

## What We Built Initially

We started from DSNO's architecture: a standard U-Net diffusion model with FNO blocks (temporal spectral convolution layers) added on top. The model takes a noisy input and outputs a discrete trajectory of N=8 denoised states at fixed, log-spaced timesteps. We then tried to enforce the consistency property on this — making the model's predictions agree across nearby noise levels along the trajectory, extending Song et al.'s convergence proofs to the neural operator setting.

## Why It Broke

The consistency training schedule uses a ratio `r/t` (where `t` is the input noise level and `r` is a nearby lower noise level) that's supposed to approach 1 over training, meaning the two comparison points get infinitesimally close. But with only 8 discrete timesteps, the maximum achievable `r/t` is about 0.22 — nowhere near 1. The scheduler math assumes continuous time, and our discrete grid fundamentally contradicts it. We'd need 50+ timesteps to get `r/t` above 0.8, and since trajectory length scales linearly with GPU memory, that's infeasible. This is also a specific instance of what ECM (Geng et al., 2024) formally identified as the **"curse of consistency"**: total error ≤ N × e_max, where increasing N can actually make things worse.

## Where We Are Now

After validating this diagnosis, we decided to move to continuous time. We did a deep research sprint across ~20 papers and converged on a direction inspired by **π-Flow** (Chen et al., ICLR 2026).

The π-Flow idea is: instead of having the network output a fixed set of discrete trajectory points, have it output a parametric representation of the entire velocity field over a time interval. One network evaluation produces this compact representation, and then you can cheaply evaluate it at any time you want — no additional network calls needed. π-Flow uses Gaussian mixture parameters for this representation; our angle is to use Fourier/spectral coefficients instead, which is more natural given DSNO's empirical finding that denoising trajectories have most of their energy concentrated in low-frequency temporal modes.

Concretely, the architecture would be: an encoder `B(x_t, t)` takes a noisy input and produces a latent trajectory code `m_t`, and a lightweight decoder `P(m_t, t, s)` takes that code plus any query time `s` and produces the state `x_s`. The encoder is expensive (one full network forward pass), the decoder is cheap (just evaluating a parametric function). Training would use ECM-style progressive tightening (start as a standard diffusion model, gradually enforce consistency) with TrigFlow parameterization (from sCM) for stable continuous-time training, and a flow matching anchor loss (from FACM) for stability.

The most recent idea on the table is an additional twist: using an autoregressive generative model as the inner velocity predictor, with diffusion/flow matching remaining as the outer loop. The autoregressive model would generate the velocity field spatially (patch by patch or token by token), which can be viewed as a function-of-space mapping — essentially a neural operator by another name. This keeps diffusion as the framework while using a fundamentally different and potentially more expressive architecture for the velocity prediction itself.

We're currently deciding between these variants and preparing to prototype on CIFAR-10.

## Relevant Papers

### Core to Our Direction

- **π-Flow: Policy-Based Few-Step Generation via Imitation Distillation** — arXiv 2510.14974
- **Consistency Trajectory Models (CTM)** — arXiv 2310.02279
- **Transition Matching Distillation (TMD)** — arXiv 2601.09881
- **Flow Map Matching** — arXiv 2406.07507

### Consistency Model Evolution

- **Consistency Models (CM)** — arXiv 2303.01469
- **Improved Consistency Training (iCT)** — arXiv 2310.14189
- **Consistency Models Made Easy (ECM)** — arXiv 2406.14548
- **Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models (sCM/TrigFlow)** — arXiv 2410.11081
- **Flow-Anchored Consistency Models (FACM)** — arXiv 2507.03738
- **Consistency Flow Matching** — arXiv 2407.02398
- **Score-Regularized Continuous-Time Consistency (rCM)** — arXiv 2510.08431

### Neural Operators + Diffusion

- **DSNO: Fast Sampling of Diffusion Models via Operator Learning** — arXiv 2211.13449
- **DiffFNO: Diffusion Fourier Neural Operator** — arXiv 2411.09911
- **Data-Driven Stochastic Closure Modeling via Conditional Diffusion Model and Neural Operator** — arXiv 2408.02965
- **NOIR: Neural Operator Mapping for Implicit Representations** — arXiv 2603.13118
- **Score-Based Diffusion Models in Function Space (DDO)** — arXiv 2302.07400

### Neural Operator Foundations

- **Neural Operator: Learning Maps Between Function Spaces** — arXiv 2108.08481
- **Fourier Neural Operators Explained** — arXiv 2512.01421
- **FC-PINO: High Precision Physics-Informed Neural Operators via Fourier Continuation** — arXiv 2211.15960
