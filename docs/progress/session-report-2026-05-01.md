# Schedule-Agnostic GMFlow Posterior — Technical Summary

**May 2026**

---

## Connection to Idea 1

Idea 1 replaces the GMFlow Gaussian-mixture policy with a boundary-constrained Fourier parameterization:

```
f(t, ξ) = φ₀(t)·x_start + φ₁(t)·x_end + Σ[ aₘ(ξ)·sin(mπt) + bₘ(ξ)·(cos(mπt) − 1) ]
```

This is a **VP / cosine schedule by construction** — the natural schedule is α = cos(πt/2), σ = sin(πt/2). The pre-existing `gmflow_posterior_mean_jit` hardcodes α = 1 − σ (linear/VE schedule) and cannot evaluate the Fourier policy.

Making the posterior mean **schedule-agnostic** is the foundational infrastructure change required before Idea 1 (or Idea 2) can be plugged into the existing Pi-Flow pipeline at any call site.

`gmflow_posterior_mean_jit_general` (Phase 1, already shipped) accepts explicit (α, σ) pairs. The work described here adds the production guards that make it safe to opt in.

---

## Summary

| Metric | Value |
|---|---|
| Work items completed | C1, C2, C3, C4 |
| Verification gates green | 7 / 8 |
| Pending | Gate (g) real-checkpoint run |

---

## Completed Work

| Item | Description | Status |
|---|---|---|
| **C1 — zeta_max clamp** | Adds `zeta_max: float = inf` to `gmflow_posterior_mean_jit_general`. Wrapper computes `finfo.max / 10.0` and passes it. Prevents float32 overflow at extreme VP ratios (e.g. t → 0). | Done |
| **C2 — gm_vars shape assert** | `torch._assert` checks on `gm_vars.size(gm_dim)==1` and `gm_vars.size(channel_dim)==1` in both JITs. The derivation only holds when var_k is shared across K components. | Done |
| **C3 — XOR dispatch gate** | Replaces silent `1−σ` auto-fill with `ValueError("Pass both … or neither.")`. No caller passes alpha kwargs, so blast radius is zero. | Done |
| **C4 — Policy schedule-lock doc** | `piflow_policies/gmflow.py` calls the legacy JIT directly (schedule-locked by design); import-level comment + inline notes at both call sites. | Done |

---

## Files Changed

### Core library

| File | Changes |
|---|---|
| `lakonlab/models/diffusions/gmflow.py` | C1: `zeta_max` param + clamp in `jit_general`; wrapper computes & passes `zeta_max` · C2: `torch._assert` on `gm_vars` shape in both JITs · C3: `ValueError` on half-pair alpha args |
| `lakonlab/models/diffusions/piflow_policies/gmflow.py` | C4: import-level comment + inline `"linear schedule only — see C4"` notes at both direct JIT calls |

### Evidence and derivations

| File | Role | Status |
|---|---|---|
| `_test_vp_stability.py` | VP stability: loads production function, guards on `has_zeta_max` | Modified |
| `_test_wrapper_dispatch.py` | Dispatch and XOR gate verification | Modified |
| `_test_gm_vars_shape_assert.py` | Shape assertion: happy path + malformed shapes | New |
| `_test_production_vp_finite.py` | Sweeps t, var_k, s_src on production JIT with `zeta_max=inf` | New |
| `_test_vp_var1_boundary.py` | `var_k=1` boundary characterization; hard assertions pass | New |
| `_test_gate_g_checkpoint.py` | Real-checkpoint equivalence harness (run pending) | New |
| `03_rollout_plan.md` | Status and verification record | Modified |
| `01_posterior_rederivation.ipynb` | Re-executed: `analytic=brute=1.913505`, `\|diff\|=1.77e-07` | Modified |

---

## Verification Results

| Gate | Description | Result | Notes |
|---|---|---|---|
| V1 | Notebook re-execution | **PASS** | `\|diff\|=1.77e-07 ≤ 5e-7` |
| V2 | Mathematica cross-check | **PASS** | SymPy equivalent confirms `missingTerm=0`, `μk ∉ free_symbols` |
| V3 | Bit-exact equivalence | **PASS** | `max abs diff = 0.0` all shapes/dtypes |
| V4 | Wrapper dispatch + audit | **PASS** | Audit clean; `0.0` both routes; XOR raises; VP smoke finite |
| V5 | Production VP-finite | **PASS** | All (s, var_k, t) combinations finite |
| V6 | SHA verification | **PASS** | Confirmed at close |
| V7 | Call-site audit | **PASS** | 3 call sites; all through wrapper; no alpha kwargs |
| V8 | Script parity | **PASS** | Both scripts exit 0; `\|diff\|=1.77e-07` |

---

## Remaining Gate

| Gate | Condition | Status |
|---|---|---|
| (g) | Real-checkpoint linear-schedule equivalence run | **Pending** — harness written; requires GPU + checkpoint access |

---

## What's Next

| Action | Unblocks |
|---|---|
| Run gate (g) harness (record then verify) | Schedule-aware sampler opt-in |
| First opt-in: add `alpha_t_src`/`alpha_t` to `pipeline_gmdit` research path | VP schedule live at research path |
| Idea 1 proper: network outputs Fourier coefficients `{aₘ, bₘ}` | Full Idea 1 |
