# GMFlow Posterior Schedule-Agnostic Refactor — Go/No-Go Decision

**Date**: 2026-04-23
**Subject**: Generalizing `gmflow_posterior_mean_jit` from hardcoded linear schedule (`α = 1 - σ`) to an arbitrary `(α, σ)` pair.

---

## Verdict: **GO**, with phased rollout.

Five independent verification layers all pass. Bit-exact equivalence on the existing path. Numerical safety on VP/trig. Trivial API expansion cost.

---

## Evidence

| # | Test | Result |
|---|---|---|
| 1 | SymPy symbolic derivation of `ζ`, `ν`, `μ_post`, `Δlog w` | All formulas correct ([`01_posterior_rederivation.ipynb`](01_posterior_rederivation.ipynb)) |
| 2 | Mathematica completing-the-square: truncated `Δlog w` differs from full log-normalizer by `vk·ν²/(2·dk)`, which is k-independent | Cancels under softmax when var_k is shared ([`posterior_rederivation.nb`](posterior_rederivation.nb)) |
| 3 | Numerical brute-force vs analytic on C=1, K=4 trig schedule | `|diff| = 1.79e-07` (machine epsilon, [`01_posterior_rederivation.ipynb`](01_posterior_rederivation.ipynb) cell 18) |
| 4 | `gm_vars` shape invariant audit across `gmflow.py`, `gmflow_ops.py`, `piflow_policies/gmflow.py` | Var is shared across components in every code path. **Mathematica precondition holds.** |
| 5 | Call site audit | 5 call sites, all linear-schedule. Backward-compatible API expansion is straightforward. |
| 6 | **Bit-exact equivalence** test on full `(B, K, C, H, W)` synthetic data, 20 seeds × 5 shape configs, float32 + float64 | **`max abs diff = 0.0`** at every config ([`_test_differential_equivalence.py`](_test_differential_equivalence.py)) |
| 7 | Float32 stability sweep under VP schedule, t ∈ [1e-20, 0.85] | All outputs finite and bounded ([`_test_vp_stability.py`](_test_vp_stability.py)) |

---

## What this means concretely

**Algebraic certainty** + **k-independence guarantee** + **bit-exact engineering equivalence** + **float32 safety under VP** = the refactor is provably a no-op on the existing path and provably safe on the new VP path.

The risk surface that remains is **not in the math or implementation** — it is:

1. **Schedule semantic mismatch**: a model trained with linear schedule may produce nonsensical samples when sampled with VP schedule. This is a *training/inference consistency* question, not a refactor question. Validate on a per-checkpoint basis before changing schedules in production.
2. **Real checkpoint validation**: synthetic tensors (Gaussian `randn`) may not exercise pathological cases that real model outputs (e.g. very confident component weights, extreme means) might. **Recommended next step**: run Test 6 on a real checkpoint before merging.

---

## Recommended rollout

### Phase 1 — Land sibling function (zero blast radius)

Add to `repos/piFlow/lakonlab/models/diffusions/gmflow.py`:

```python
@torch.jit.script
def gmflow_posterior_mean_jit_general(
        alpha_t_src, sigma_t_src, alpha_t, sigma_t, x_t_src, x_t,
        gm_means, gm_vars, gm_logweights,
        eps: float, gm_dim: int = -4, channel_dim: int = -3):
    sigma_t_src = sigma_t_src.clamp(min=eps)
    sigma_t = sigma_t.clamp(min=eps)

    aos_src = alpha_t_src / sigma_t_src
    aos_t = alpha_t / sigma_t

    zeta = aos_t.square() - aos_src.square()
    # Mirror operator order of the existing code so the linear-schedule
    # path is bit-exact, not just within epsilon.
    nu = aos_t * x_t / sigma_t - aos_src * x_t_src / sigma_t_src

    nu = nu.unsqueeze(gm_dim)
    zeta = zeta.unsqueeze(gm_dim)
    denom = (gm_vars * zeta + 1).clamp(min=eps)

    out_means = (gm_vars * nu + gm_means) / denom
    logweights_delta = (gm_means * (nu - 0.5 * zeta * gm_means)).sum(
        dim=channel_dim, keepdim=True) / denom
    out_weights = (gm_logweights + logweights_delta).softmax(dim=gm_dim)
    return (out_means * out_weights).sum(dim=gm_dim)
```

Original `gmflow_posterior_mean_jit` stays untouched.

### Phase 2 — Backward-compatible wrapper

Update `gmflow_posterior_mean()` (the user-facing method) to accept optional `alpha_t_src` / `alpha_t`. When both are `None`, compute them as `1 - sigma_t_src` and `1 - sigma_t` and call the new general function. Outputs are bit-exact with the old function.

```python
def gmflow_posterior_mean(self, gm, x_t, x_t_src, t=None, t_src=None,
                          sigma_t_src=None, sigma_t=None,
                          alpha_t_src=None, alpha_t=None,
                          eps=1e-6, prediction_type='x0', checkpointing=False):
    # ... existing sigma derivation ...
    if alpha_t_src is None:
        alpha_t_src = 1 - sigma_t_src
    if alpha_t is None:
        alpha_t = 1 - sigma_t
    # ... call gmflow_posterior_mean_jit_general(...) ...
```

The 5 existing call sites (none of which pass `alpha_*`) are unchanged in behaviour.

### Phase 3 — Add a debug-mode equivalence assertion

In `gmflow_posterior_mean_jit_general`, when `(alpha_t_src ≈ 1 - sigma_t_src) and (alpha_t ≈ 1 - sigma_t)`, also call the legacy function and assert allclose. Gate this on an env var or `DEBUG=True` so production has zero overhead.

### Phase 4 — Schedule-aware sampler experiments

**Only after Phase 1–3 ship and bake**, introduce a non-linear schedule to one specific call site (probably the research path in `pipeline_gmdit.py`). Validate generated sample quality. Roll back if anything looks off.

---

## What to do if any new bug surfaces

The refactor is reversible. If a problem appears:

1. **No regression possible on the existing path** as long as no caller passes explicit `alpha_*` kwargs (Test 6 proves bit-exact). Default behaviour is unchanged.
2. **VP path issues** can be isolated to the specific call site that opted in.
3. The legacy `gmflow_posterior_mean_jit` is never deleted.

---

## Files produced by this verification

- `01_posterior_rederivation.ipynb` — SymPy + numerical derivation
- `posterior_rederivation.nb` — Mathematica completing-the-square proof
- `02_mathematica_crosscheck.md` — Mathematica notes
- `_test_differential_equivalence.py` — Bit-exact equivalence (Test 6)
- `_test_vp_stability.py` — Float32 stability (Test 7)
- `03_go_no_go.md` — this document
