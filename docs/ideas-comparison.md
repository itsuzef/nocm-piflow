# Ideas Comparison: Paper (Sec 7) vs Our Approach

**Date:** Apr 15, 2026

The paper's Section 7 and our worknotes converge on the same high-level directions. This document maps where they align, where we go further, and what is unique to our approach.

---

## Side-by-Side Overview

| Direction | Paper (Sec 7) | Our Approach |
|---|---|---|
| Replace GMM with trig/Fourier policy | Sketched — sin/cos basis, boundary terms only | Full Fourier series expansion with M harmonics + rigorous posterior re-derivation |
| Replace GMM with PDE-defined policy | Spatial PDE coefficients $k_1(\xi), k_2(\xi)$ | Same concept — open questions on scalar vs spatial, stability (not yet examined) |
| NO as policy generator | Listed as one of three ideas | Already framed as umbrella in `pi_flow_three_ideas.md` — open questions on FNO interface (not yet examined) |
| Latent embedding ($B_\phi + P_\theta$) | Present (subsection) | Not pursued — ruled out as unclear benefit |
| Schedule-agnostic posterior | Not addressed | **Core technical contribution** — full re-derivation (see below) |

---

## Idea 1 — Trig / Fourier Policy

### What the paper has

$$\Psi_\Pi(t, \xi) = \phi_0(t)\, x_{start}(\xi) + \phi_1(t)\, x_{stop}(\xi) + \sum_{m=1}^M a_m(\xi)\sin(m\pi t) + b_m(\xi)(\cos(m\pi t) - 1)$$

Boundary terms $\phi_0, \phi_1$ enforce endpoint conditions. The network outputs $\{a_m, b_m\}$.

### What we add

**1. Full Fourier series expansion is already in the paper formula** — we use the same form.

**2. The critical gap the paper doesn't address:** GMFlow's posterior update (`gmflow_posterior_mean_jit`) hardcodes `alpha = 1 - sigma` (linear schedule, variance-exploding). Trig schedule uses `alpha = cos(πt/2)`, `sigma = sin(πt/2)` — a variance-preserving (VP) process where `alpha² + sigma² = 1`. The entire posterior derivation changes.



**General-form posterior (schedule-agnostic):**

| Term | Formula |
|---|---|
| Precision gain | $\zeta = (\alpha_t / \sigma_t)^2 - (\alpha_s / \sigma_s)^2$ |
| Info vector gain | $\nu = \alpha_t x_t / \sigma_t^2 - \alpha_s x_s / \sigma_s^2$ |
| Per-component denom | $d_k = \text{var}_k \cdot \zeta + 1$ |
| Posterior mean | $\bar{\mu}_k = (\text{var}_k \cdot \nu + \mu_k) / d_k$ |
| Log weight delta | $\Delta \log w_k = \mu_k^T(\nu - 0.5\,\zeta\,\mu_k) / d_k$ |

Under linear schedule (`alpha = 1 - sigma`) this reduces to the existing code. Under trig it gives a new formula that the paper doesn't derive.

**3. VP vs VE distinction has numerical consequences.** Under linear (VE), SNR = `(1-σ)²/σ²` which has a gentle slope. Under trig (VP), SNR = `cot²(πt/2)` which blows up near `t = 0`. The existing `clamp(min=eps)` guards may be insufficient — part of the edge stability analysis in the workspace.

---

## Idea 2 — PDE Parameter Policy

> **Not yet examined in depth. Notes below are from preliminary docs (`pi_flow_three_ideas.md`, worklog) — open questions, not settled positions.**

### What the paper has

The paper's sec 7 already proposes spatially-varying PDE coefficients:
$$\partial_t X = a_\lambda(\xi)\,\Delta X + b_\lambda(\xi)\cdot\nabla X + c_\lambda(\xi)\,X + f_\lambda(t, \xi)$$
The 2nd-order (damped oscillation) version from the whiteboard:
$$\partial_{tt} X = -k_1(\xi)(X - \hat{X}_\infty(\xi)) - k_2(\xi)\,\partial_t X + u(t, \xi)$$

Note: the paper already uses $k_1(\xi)$ (spatial maps), not scalars. The scalar framing is a simplification in the worklog session plan, not the paper's actual proposal.

### Open questions (from worklog — not yet resolved)

- **Scalar vs spatial:** scalar params shared across all pixels probably not expressive enough for $128\times128$. But spatial maps bring output dimensionality close to Fourier territory — is there a real distinction?
- **Stability:** $k_1 > 0$ causes ODE solution to diverge. Constrained output needed.
- **Relationship to Idea 1:** per-pixel ODE with spatially varying coefficients may be mathematically similar to a low-order Fourier expansion. Not yet worked out.

---

## Idea 3 — Neural Operator as Policy Generator

> **Not yet examined in depth. The hierarchy framing below is already stated explicitly in `pi_flow_three_ideas.md` — not original to this doc.**

### What the paper has

Listed as one of three parallel ideas:
$$G_\theta(x_t, t) = \mathcal{N}_\theta(f(x_t, t)) = \Psi_\Pi(x_t, t)$$

### From `pi_flow_three_ideas.md` (already established)

Idea 3 is the umbrella; Ideas 1 and 2 are instantiations:
```
Idea 3: NO generates policy code m_r from (x_r, r)
├── Idea 1: m_r = {a_m(ξ), b_m(ξ)} — Fourier coefficients
└── Idea 2: m_r = {k_1(ξ), k_2(ξ), f(ξ)} — PDE parameters
```

**Caveat from that doc:** don't overstate NO capabilities. You still need a finite representation, cheap query mechanism, and stable training — otherwise you've replaced one constraint with a harder optimization problem.

### Open questions (from worklog — not yet resolved)

- How does FNO from `repos/neuraloperator` interface with PiFlow's policy contract (`(x_t, sigma_t) → velocity`)?
- Does wrapping in a NO add anything over a bigger backbone? PiFlow's policy is already per-pixel resolution-aware — the resolution-invariance benefit of NO may not apply.

---

## What Is Unique to Our Work

1. **Schedule-agnostic GMFlow posterior** — full information-form re-derivation for arbitrary `(alpha_t, sigma_t)`. Paper assumes linear; we make it work for trig/VP. Verified symbolically (SymPy + Mathematica) and numerically (PyTorch). See `math_derivation_workspace.md`.

2. **VP vs VE analysis** — the existing codebase is VE throughout. Switching to trig (VP) changes the posterior, the loss, the SNR curve, and the numerical stability story. Paper doesn't address this.

3. *(Ideas 2 and 3 — unique contributions TBD once examined in depth.)*
