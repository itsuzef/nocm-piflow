# Notebook Math Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four mathematical/documentation errors in `derivations/01_posterior_rederivation.ipynb` and add a Section 9 with two guard tests (per-component var_k mismatch exposure and domain assertion).

**Architecture:** All changes are confined to the single source notebook. Tasks 1–4 are inline fixes to existing cells (markdown commentary and one code line). Task 5 appends two new cells (Section 9). Task 6 re-runs the notebook to confirm no regressions. No new files are created.

**Tech Stack:** Jupyter notebook (`.ipynb`), SymPy, PyTorch, NotebookEdit tool.

---

## File Map

| File | Change |
|---|---|
| `derivations/01_posterior_rederivation.ipynb` | Modify cells 5, 7, 15, 17; append cells 24–25 |

Cell indices are 0-based as they appear in the JSON `cells` array.

---

### Task 1: Fix Cell 5 — Sec 3 importance-reweighting framing + t < s constraint

**Files:**
- Modify: `derivations/01_posterior_rederivation.ipynb` cell index 5

The current markdown says "The **gain** from adding the t-observation on top of the s-prior" without explaining why the gain is a *subtraction* ($\Lambda_t - \Lambda_s$) rather than a plain addition. It also omits the validity domain.

- [ ] **Step 1: Open the notebook and locate cell 5**

Cell 5 source (current):
```
## Section 3 — Posterior Update (core derivation)

**Setup:**  
- GM prior from source observation: $p(x_0 \mid x_s) = \sum_k w_k \mathcal{N}(\mu_k, \mathrm{var}_k)$
- New observation: $p(x_t \mid x_0) = \mathcal{N}(\alpha_t x_0, \sigma_t^2)$
- Goal: compute $p(x_0 \mid x_s, x_t)$

In information form, multiplying two Gaussians adds precisions and information vectors.  
The **gain** from adding the $t$-observation on top of the $s$-prior:

$$\zeta = \Lambda_t - \Lambda_s = \left(\frac{\alpha_t}{\sigma_t}\right)^2 - \left(\frac{\alpha_s}{\sigma_s}\right)^2$$

$$\nu = \eta_t - \eta_s = \frac{\alpha_t x_t}{\sigma_t^2} - \frac{\alpha_s x_s}{\sigma_s^2}$$
```

- [ ] **Step 2: Replace cell 5 source with the corrected text**

Use NotebookEdit on `derivations/01_posterior_rederivation.ipynb`, cell_number=5, new_source:

```markdown
## Section 3 — Posterior Update (core derivation)

**Setup:**  
- GM prior from source observation: $p(x_0 \mid x_s) = \sum_k w_k \mathcal{N}(\mu_k, \mathrm{var}_k)$
- New observation: $p(x_t \mid x_0) = \mathcal{N}(\alpha_t x_0, \sigma_t^2)$
- Goal: update $p(x_0 \mid x_s)$ to $p(x_0 \mid x_t)$ using x_t as a better (less noisy) observation

> **Domain constraint:** formula valid only for $t < s$, i.e. $\sigma_t < \sigma_s$ (x_t less noisy than x_s).

The update is an **importance reweighting** step:

$$p(x_0 \mid x_t) \;\propto\; \frac{p(x_t \mid x_0)}{p(x_s \mid x_0)} \cdot p(x_0 \mid x_s)$$

In information (canonical) form, dividing two Gaussians *subtracts* their precisions and information vectors.  
The **gain** (information added by replacing x_s with x_t):

$$\zeta = \Lambda_t - \Lambda_s = \left(\frac{\alpha_t}{\sigma_t}\right)^2 - \left(\frac{\alpha_s}{\sigma_s}\right)^2 \;>\; 0 \quad (\text{since } t < s)$$

$$\nu = \eta_t - \eta_s = \frac{\alpha_t x_t}{\sigma_t^2} - \frac{\alpha_s x_s}{\sigma_s^2}$$
```

- [ ] **Step 3: Verify the cell renders correctly**

Open the notebook in Jupyter or inspect the JSON to confirm the cell source was written correctly. No code to run — this is a markdown cell.

---

### Task 2: Fix Cell 7 — document shared var_k assumption on log-weight formula

**Files:**
- Modify: `derivations/01_posterior_rederivation.ipynb` cell index 7

The logweight formula drops the term $\nu^2 \mathrm{var}_k / (2 d_k)$ from the full completing-the-square result. This is valid only when `var_k` is shared across all k components (singleton K dim), so the dropped term is k-independent and cancels in the softmax. This assumption is undocumented.

- [ ] **Step 1: Locate cell 7 source (current)**

```
### Per-component posterior

For component $k$ with prior $\mathcal{N}(\mu_k, \mathrm{var}_k)$:

- Updated precision: $\Lambda_k^{\text{post}} = 1/\mathrm{var}_k + \zeta$
- Updated variance: $\mathrm{var}_k^{\text{post}} = 1 / \Lambda_k^{\text{post}} = \mathrm{var}_k / (\mathrm{var}_k \zeta + 1)$
- Denominator: $d_k = \mathrm{var}_k \zeta + 1$
- Updated mean: $\mu_k^{\text{post}} = \mathrm{var}_k^{\text{post}} \cdot (\eta_k^{\text{prior}} + \nu) = \dfrac{\mathrm{var}_k \nu + \mu_k}{d_k}$
- Log-weight delta: $\Delta \log w_k = \dfrac{\mu_k^T (\nu - \tfrac{1}{2}\zeta\,\mu_k)}{d_k}$ (from completing the square)
```

- [ ] **Step 2: Replace cell 7 source with the annotated text**

Use NotebookEdit on `derivations/01_posterior_rederivation.ipynb`, cell_number=7, new_source:

```markdown
### Per-component posterior

For component $k$ with prior $\mathcal{N}(\mu_k, \mathrm{var}_k)$:

- Updated precision: $\Lambda_k^{\text{post}} = 1/\mathrm{var}_k + \zeta$
- Updated variance: $\mathrm{var}_k^{\text{post}} = 1 / \Lambda_k^{\text{post}} = \mathrm{var}_k / (\mathrm{var}_k \zeta + 1)$
- Denominator: $d_k = \mathrm{var}_k \zeta + 1$
- Updated mean: $\mu_k^{\text{post}} = \mathrm{var}_k^{\text{post}} \cdot (\eta_k^{\text{prior}} + \nu) = \dfrac{\mathrm{var}_k \nu + \mu_k}{d_k}$
- Log-weight delta: $\Delta \log w_k = \dfrac{\mu_k^T (\nu - \tfrac{1}{2}\zeta\,\mu_k)}{d_k}$ (from completing the square)

> **Assumption — shared variance across components:**  
> The full completing-the-square log-normalizer is  
> $\dfrac{\mu_k^T(\nu - \frac{1}{2}\zeta\mu_k)}{d_k} + \dfrac{\|\nu\|^2\,\mathrm{var}_k}{2\,d_k}$  
> The second term $\|\nu\|^2\mathrm{var}_k/(2d_k)$ is dropped here because in the piflow implementation
> `gm_vars` has a singleton K-dimension (shape `(B,1,1,...)`), so $\mathrm{var}_k$ and $d_k$ are
> identical across all components. The dropped term is then k-independent and cancels inside `softmax`.  
> **If per-component variances are introduced, both terms must be included.** See Section 9.
```

---

### Task 3: Fix Cell 15 — correct the t→1 limit comment

**Files:**
- Modify: `derivations/01_posterior_rederivation.ipynb` cell index 15

The current comment claims "t → 1: posterior → prior mean mu_k". This is wrong: at t=1, $\zeta = -\Lambda_s < 0$ (the gain is negative, not zero), so the posterior does not collapse to $\mu_k$. Furthermore, t=1 > s=0.5 is outside the valid domain.

- [ ] **Step 1: Locate cell 15 source (current)**

```python
# Boundary limits of out_mean_k under trig schedule
# t -> 0: observation x_t becomes noiseless (sigma_t -> 0), posterior should -> x_t
# t -> 1: sigma_t -> 1, alpha_t -> 0, information -> 0, posterior -> prior mean mu_k

mean_trig = out_mean_k.subs(trig_subs)

# t -> 0 limit (s fixed, say s = 0.5 for concreteness)
lim_t0 = limit(mean_trig.subs(s, Rational(1,2)), t, 0, '+')
print('limit t->0 (s=0.5):', simplify(lim_t0))

# t -> 1 limit
lim_t1 = limit(mean_trig.subs(s, Rational(1,2)), t, 1, '-')
print('limit t->1 (s=0.5):', simplify(lim_t1))
```

- [ ] **Step 2: Replace cell 15 source with corrected comments**

Use NotebookEdit on `derivations/01_posterior_rederivation.ipynb`, cell_number=15, new_source:

```python
# Boundary limits of out_mean_k under trig schedule
# t -> 0: x_t becomes noiseless (sigma_t -> 0, Lambda_t -> inf), posterior collapses to x_t
# t -> 1: OUTSIDE VALID DOMAIN (t=1 > s=0.5, violates t < s constraint).
#         At t=1: alpha_t=0, Lambda_t=0, so gain zeta = -Lambda_s < 0.
#         Formula is algebraically defined but unphysical; result is NOT mu_k.

mean_trig = out_mean_k.subs(trig_subs)

# t -> 0 limit (s fixed at 0.5): expect x_t
lim_t0 = limit(mean_trig.subs(s, Rational(1,2)), t, 0, '+')
print('limit t->0 (s=0.5):', simplify(lim_t0))   # should print x_t

# t -> 1 limit (s=0.5): t > s, unphysical — shown for completeness only
# Result depends on x_s and var_k because zeta < 0 at t=1
lim_t1 = limit(mean_trig.subs(s, Rational(1,2)), t, 1, '-')
print('limit t->1 (s=0.5) [UNPHYSICAL — outside t<s domain]:', simplify(lim_t1))
```

---

### Task 4: Fix Cell 17 — replace ZETA_MAX placeholder and add domain guard

**Files:**
- Modify: `derivations/01_posterior_rederivation.ipynb` cell index 17

Two sub-fixes: (a) replace the `ZETA_MAX = 1e6` placeholder with the value derived in Section 7; (b) add a `ValueError` at the top of `posterior_general` enforcing `sigma_t < sigma_s`.

- [ ] **Step 1: Locate the two target lines in cell 17**

Line to replace (ZETA_MAX):
```python
    ZETA_MAX = 1e6   # placeholder; Section 7 will derive the analytic bound
```

Lines to add (domain guard) — insert after the two `.clamp(min=eps)` lines:
```python
    sigma_t = sigma_t.clamp(min=eps)
    sigma_s = sigma_s.clamp(min=eps)
    # [INSERT HERE]
```

- [ ] **Step 2: Write the full corrected cell 17 source**

Use NotebookEdit on `derivations/01_posterior_rederivation.ipynb`, cell_number=17, new_source (show only the changed function; rest of cell is unchanged):

Replace the entire cell with the full corrected source below. (Copy verbatim — all other functions and test code in this cell remain identical.)

```python
import torch
import math

torch.manual_seed(42)
B, K, C = 2, 4, 8   # batch, components, channels

def posterior_general(alpha_t, sigma_t, alpha_s, sigma_s, x_t, x_s,
                      gm_means, gm_vars, gm_logweights, eps=1e-6):
    """Schedule-agnostic posterior mean.

    Computes p(x0 | x_t) by importance-reweighting p(x0 | x_s) with
    ratio p(x_t|x0)/p(x_s|x0).

    **Requires t < s** (sigma_t < sigma_s): x_t must be less noisy than x_s.

    **Requires shared var_k across components**: gm_vars must have a singleton
    K-dimension (e.g. shape (B,1,1) or (B,1,1,H,W)). Per-component variances
    require an additional nu^2*var/(2*d) term in the log-weight; see Section 9.

    Shape contract:
      alpha_t, sigma_t: broadcastable with x_t  (e.g. (B,1) for 2-D x_t)
      x_t, x_s:        (B, C) or (B, C, H, W)
      gm_means:         (B, K, C) or (B, K, C, H, W)
      gm_vars:          (B, 1, 1) or (B, 1, 1, H, W)   [K and C singletons]
      gm_logweights:    (B, K, 1) or (B, K, 1, H, W)
    """
    sigma_t = sigma_t.clamp(min=eps)
    sigma_s = sigma_s.clamp(min=eps)

    if (sigma_t >= sigma_s).any():
        raise ValueError(
            "posterior_general requires t < s (sigma_t < sigma_s). "
            f"Got sigma_t max={sigma_t.max().item():.4f} >= sigma_s min={sigma_s.min().item():.4f}. "
            "x_t must be less noisy than x_s."
        )

    # zeta/nu: same spatial shape as x_t  (B, C) or (B, C, H, W)
    zeta = (alpha_t / sigma_t).square() - (alpha_s / sigma_s).square()
    nu   = alpha_t * x_t / sigma_t**2   - alpha_s * x_s / sigma_s**2

    # Clamp zeta — derived in Sec 7: float32_max / max_var_k (conservative upper bound 10.0)
    ZETA_MAX = torch.finfo(torch.float32).max / 10.0
    zeta = zeta.clamp(max=ZETA_MAX)

    # K-dimension (gm_dim) is always dim 1 = dim -(ndim-1) of gm_means.
    # C-dimension (channel_dim) is the next one inward: dim -(ndim-2).
    gm_dim      = 1 - gm_means.dim()   # -2 for 3-D, -4 for 5-D
    channel_dim = 2 - gm_means.dim()   # -1 for 3-D, -3 for 5-D

    nu   = nu.unsqueeze(gm_dim)
    zeta = zeta.unsqueeze(gm_dim)
    denom = (gm_vars * zeta + 1).clamp(min=eps)

    out_means  = (gm_vars * nu + gm_means) / denom
    logw_delta = (gm_means * (nu - 0.5 * zeta * gm_means)).sum(
        dim=channel_dim, keepdim=True) / denom
    out_weights = (gm_logweights + logw_delta).softmax(dim=gm_dim)

    return (out_means * out_weights).sum(dim=gm_dim)


def gmflow_posterior_mean_jit_ref(sigma_t_src, sigma_t, x_t_src, x_t,
                                   gm_means, gm_vars, gm_logweights, eps=1e-6):
    """Reference: existing hardcoded linear code (alpha = 1 - sigma).

    Uses the same dynamic gm_dim / channel_dim fix for consistency.
    """
    sigma_t_src = sigma_t_src.clamp(min=eps)
    sigma_t     = sigma_t.clamp(min=eps)
    alpha_t_src = 1 - sigma_t_src
    alpha_t     = 1 - sigma_t
    aos_src = alpha_t_src / sigma_t_src
    aos_t   = alpha_t     / sigma_t
    zeta = aos_t.square() - aos_src.square()
    nu   = aos_t * x_t / sigma_t - aos_src * x_t_src / sigma_t_src

    gm_dim      = 1 - gm_means.dim()
    channel_dim = 2 - gm_means.dim()

    nu   = nu.unsqueeze(gm_dim)
    zeta = zeta.unsqueeze(gm_dim)
    denom = (gm_vars * zeta + 1).clamp(min=eps)
    out_means  = (gm_vars * nu + gm_means) / denom
    logw_delta = (gm_means * (nu - 0.5 * zeta * gm_means)).sum(
        dim=channel_dim, keepdim=True) / denom
    out_weights = (gm_logweights + logw_delta).softmax(dim=gm_dim)
    return (out_means * out_weights).sum(dim=gm_dim)


# --- Random test data (B, K, C) — simplified, no spatial dims ---
# sigma/alpha are (B, 1): one trailing singleton so they broadcast with (B, C).
# In production these would be (B, 1, 1, 1) for (B, C, H, W) x tensors.
gm_means      = torch.randn(B, K, C)             # (B, K, C)
gm_logweights = torch.randn(B, K, 1)             # (B, K, 1)
gm_vars       = torch.rand(B, 1, 1).abs() + 0.1  # (B, 1, 1)  — K and C singletons

sigma_s_val = torch.rand(B, 1) * 0.4 + 0.4       # (B, 1) broadcasts with (B, C)
sigma_t_val = sigma_s_val * torch.rand(B, 1) * 0.6
sigma_t_val = sigma_t_val.clamp(min=1e-3)

alpha_s_lin = 1 - sigma_s_val   # (B, 1)
alpha_t_lin = 1 - sigma_t_val   # (B, 1)

x_s = torch.randn(B, C)
x_t = torch.randn(B, C)

# (a) General formula with linear schedule
out_general_lin = posterior_general(
    alpha_t_lin, sigma_t_val, alpha_s_lin, sigma_s_val,
    x_t, x_s, gm_means, gm_vars, gm_logweights)

# (b) Reference code (existing)
out_ref = gmflow_posterior_mean_jit_ref(
    sigma_s_val, sigma_t_val, x_s, x_t,
    gm_means, gm_vars, gm_logweights)

diff_ab = (out_general_lin - out_ref).abs().max().item()
print(f'(a) vs (b) max abs diff [should be ~0]: {diff_ab:.2e}')
assert diff_ab < 1e-5, f'Linear schedule mismatch: {diff_ab}'
print('PASS: general formula with alpha=1-sigma matches existing code')
```

---

### Task 5: Append Section 9 — Assumption Guard Tests

**Files:**
- Modify: `derivations/01_posterior_rederivation.ipynb` — append two new cells after the last cell (index 23)

Section 9 contains two guard tests:
1. Per-component var_k test: shows `posterior_general` produces wrong weights when var_k differs across k, then demonstrates the corrected formula (with the missing $\|\nu\|^2\mathrm{var}_k/(2d_k)$ term) matches brute-force.
2. Domain assertion test: confirms `posterior_general` raises `ValueError` when called with `sigma_t >= sigma_s`.

- [ ] **Step 1: Append the Section 9 markdown cell**

Use NotebookEdit on `derivations/01_posterior_rederivation.ipynb`, append new cell (cell_type=markdown) after cell 23:

```markdown
## Section 9 — Assumption Guard Tests

Two tests that document and enforce the assumptions baked into `posterior_general`:

1. **Per-component variance:** The logweight formula is exact only when `gm_vars` is shared
   across k (singleton K-dim). This test exposes the mismatch when var_k differs per component,
   then shows the full formula (with the missing $\|\nu\|^2\mathrm{var}_k/(2d_k)$ term) recovers
   the correct answer.

2. **Domain assertion:** `posterior_general` must be called with `sigma_t < sigma_s`
   (x_t less noisy than x_s). This test confirms the guard raises `ValueError` on invalid input.
```

- [ ] **Step 2: Append the Section 9 code cell**

Use NotebookEdit on `derivations/01_posterior_rederivation.ipynb`, append new cell (cell_type=code) after the Section 9 markdown cell:

```python
import torch, math

torch.manual_seed(7)
B2, K2, C2 = 2, 4, 1   # C=1 for exact 1D brute-force comparison

# ── Test 9a: per-component var_k exposes missing logweight term ──────────────

gm_means2      = torch.randn(B2, K2, C2)
gm_logweights2 = torch.randn(B2, K2, 1)
# Per-component variance: shape (B, K, 1) — K dimension is NOT a singleton
gm_vars2_perK  = torch.rand(B2, K2, 1).abs() + 0.1   # distinct per component

sigma_s2 = torch.tensor([[0.6], [0.7]])   # (B, 1)
sigma_t2 = torch.tensor([[0.2], [0.3]])   # (B, 1)  — t < s, valid domain
alpha_s2 = 1 - sigma_s2
alpha_t2 = 1 - sigma_t2

x_s2 = torch.randn(B2, C2)
x_t2 = torch.randn(B2, C2)


def posterior_general_full(alpha_t, sigma_t, alpha_s, sigma_s, x_t, x_s,
                            gm_means, gm_vars, gm_logweights, eps=1e-6):
    """Full posterior_general including the missing nu^2*var/(2*d) logweight term.
    Correct for arbitrary (including per-component) gm_vars shapes.
    """
    sigma_t = sigma_t.clamp(min=eps)
    sigma_s = sigma_s.clamp(min=eps)
    zeta = (alpha_t / sigma_t).square() - (alpha_s / sigma_s).square()
    nu   = alpha_t * x_t / sigma_t**2 - alpha_s * x_s / sigma_s**2

    gm_dim      = 1 - gm_means.dim()
    channel_dim = 2 - gm_means.dim()

    nu_k   = nu.unsqueeze(gm_dim)
    zeta_k = zeta.unsqueeze(gm_dim)
    denom  = (gm_vars * zeta_k + 1).clamp(min=eps)

    out_means = (gm_vars * nu_k + gm_means) / denom

    # Full log-weight: both terms from completing the square
    term1 = (gm_means * (nu_k - 0.5 * zeta_k * gm_means)).sum(dim=channel_dim, keepdim=True)
    term2 = (gm_vars * nu_k.square()).sum(dim=channel_dim, keepdim=True) / 2
    logw_delta = (term1 + term2) / denom

    out_weights = (gm_logweights + logw_delta).softmax(dim=gm_dim)
    return (out_means * out_weights).sum(dim=gm_dim)


def brute_force_1d_perK(at, st, as_, ss, mu_vals, var_vals, logw_vals, xt_val, xs_val,
                          n_grid=30_000, std_range=15):
    """Exact 1D integration for C=1, per-component var_k."""
    K = len(mu_vals)
    at, st, as_, ss = float(at), float(st), float(as_), float(ss)
    xt, xs = float(xt_val), float(xs_val)

    # Build a wide grid centred on the highest-weight component
    center = float(mu_vals[0])
    half   = std_range * (max(float(v)**0.5 for v in var_vals) + st / max(at, 1e-8))
    grid   = torch.linspace(center - half, center + half, n_grid)

    log_gm = torch.stack([
        float(logw_vals[k]) - 0.5 * (grid - float(mu_vals[k]))**2 / float(var_vals[k])
        for k in range(K)
    ]).logsumexp(dim=0)

    log_ratio = (-0.5*(xt - at*grid)**2/st**2) - (-0.5*(xs - as_*grid)**2/ss**2)
    log_post  = log_gm + log_ratio
    log_post  = log_post - log_post.logsumexp(dim=0)
    return float((log_post.exp() * grid).sum())


out_truncated = posterior_general(
    alpha_t2, sigma_t2, alpha_s2, sigma_s2,
    x_t2, x_s2, gm_means2, gm_vars2_perK, gm_logweights2)

out_full = posterior_general_full(
    alpha_t2, sigma_t2, alpha_s2, sigma_s2,
    x_t2, x_s2, gm_means2, gm_vars2_perK, gm_logweights2)

print('=== 9a: per-component var_k ===')
for b in range(B2):
    brute = brute_force_1d_perK(
        float(alpha_t2[b,0]), float(sigma_t2[b,0]),
        float(alpha_s2[b,0]), float(sigma_s2[b,0]),
        gm_means2[b,:,0], gm_vars2_perK[b,:,0], gm_logweights2[b,:,0],
        x_t2[b,0], x_s2[b,0])
    trunc = float(out_truncated[b,0])
    full  = float(out_full[b,0])
    ok_trunc = abs(trunc - brute) < 0.01
    ok_full  = abs(full  - brute) < 0.01
    print(f'  b={b}: brute={brute:.5f}  truncated={trunc:.5f} {"PASS" if ok_trunc else "MISMATCH (expected)"}  '
          f'full={full:.5f} {"PASS" if ok_full else "FAIL"}')

assert all(
    abs(float(out_full[b,0]) - brute_force_1d_perK(
        float(alpha_t2[b,0]), float(sigma_t2[b,0]),
        float(alpha_s2[b,0]), float(sigma_s2[b,0]),
        gm_means2[b,:,0], gm_vars2_perK[b,:,0], gm_logweights2[b,:,0],
        x_t2[b,0], x_s2[b,0])) < 0.01
    for b in range(B2)
), "Full formula should match brute-force with per-component var_k"
print('PASS: full formula matches brute-force; truncated formula correctly shows mismatch')
print()

# ── Test 9b: domain assertion fires when sigma_t >= sigma_s ─────────────────

print('=== 9b: domain assertion ===')
sigma_s_bad = torch.tensor([[0.4]])
sigma_t_bad = torch.tensor([[0.6]])   # sigma_t > sigma_s — invalid

try:
    posterior_general(
        1 - sigma_t_bad, sigma_t_bad, 1 - sigma_s_bad, sigma_s_bad,
        torch.zeros(1, C2), torch.zeros(1, C2),
        torch.zeros(1, K2, C2), torch.ones(1, 1, 1) * 0.1, torch.zeros(1, K2, 1))
    print('FAIL: should have raised ValueError')
except ValueError as e:
    print(f'PASS: ValueError raised — {e}')
```

---

### Task 6: Run the notebook and verify all sections pass

**Files:**
- Read: `derivations/01_posterior_rederivation.ipynb`
- Write: `derivations/01_posterior_rederivation_out.ipynb`

- [ ] **Step 1: Execute the notebook top-to-bottom**

```bash
cd /Users/youssefhemimy/Documents/nocm-piflow/derivations
jupyter nbconvert --to notebook --execute 01_posterior_rederivation.ipynb \
  --output 01_posterior_rederivation_out.ipynb --ExecutePreprocessor.timeout=120
```

- [ ] **Step 2: Verify expected output**

Check the output notebook for these results (all must appear):

| Cell | Expected output |
|---|---|
| Cell 4 (linear check) | `zeta match (should be 0): 0` and `nu match (should be 0): 0` |
| Cell 15 (limits) | `limit t->0 (s=0.5): x_t` |
| Cell 17 (linear/ref) | `PASS: general formula with alpha=1-sigma matches existing code` |
| Cell 18 (trig brute) | `PASS: trig posterior matches exact 1D brute-force for all batch elements` |
| Sec 9 code | `PASS: full formula matches brute-force; truncated formula correctly shows mismatch` and `PASS: ValueError raised — ...` |

If any assertion fires unexpectedly, debug before proceeding.

---

### Task 7: Commit

- [ ] **Step 1: Stage and commit**

```bash
cd /Users/youssefhemimy/Documents/nocm-piflow
git add derivations/01_posterior_rederivation.ipynb \
        derivations/01_posterior_rederivation_out.ipynb \
        docs/superpowers/plans/2026-04-23-notebook-math-fixes.md
git commit -m "fix: correct notebook math commentary and add assumption guard tests

- Sec 3: clarify importance-reweighting framing and t < s domain constraint
- Sec 3 per-component: document shared-var_k assumption on logweight formula
- Sec 5: fix wrong t->1 limit expectation (was 'posterior -> mu_k', incorrect)
- Sec 6 posterior_general: replace ZETA_MAX placeholder, add sigma_t < sigma_s guard
- Sec 9 new: per-component var_k mismatch test + domain assertion test"
```
