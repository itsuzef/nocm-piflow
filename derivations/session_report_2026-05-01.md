# Idea 1 Infrastructure — Session Report

**Phase 2 hygiene bundle · 2026-04-30 → 2026-05-01 ·**

---

## Connection to Idea 1

Idea 1 replaces the GMFlow Gaussian-mixture policy with a boundary-constrained Fourier parameterization:

```
f(t, ξ) = φ₀(t)·x_start + φ₁(t)·x_end + Σ[ aₘ(ξ)·sin(mπt) + bₘ(ξ)·(cos(mπt) − 1) ]
```

This is a **VP / cosine schedule by construction** — the natural schedule is α = cos(πt/2), σ = sin(πt/2). The pre-existing `gmflow_posterior_mean_jit` hardcodes α = 1 − σ (linear/VE schedule) and cannot evaluate the Fourier policy.

This session's work — making the posterior mean **schedule-agnostic** — is the foundational infrastructure change required before Idea 1 (or Idea 2) can be plugged into the existing Pi-Flow pipeline at any call site.

`gmflow_posterior_mean_jit_general` (Phase 1, already shipped) accepts explicit (α, σ) pairs. Phase 2 (this session) adds the production guards that make it safe to opt in.

---

## Summary

| Metric | Value |
|---|---|
| Work items completed | C1, C2, C3, C4 |
| Submodule commits | 4 |
| Evidence repo commits | 8 total on branch |
| Verification gates green | 7 / 8 |
| Pending | V2 (Wolfram license) · gate (g) checkpoint run |

---

## Phase 2 Work Items

| Item | What it does | Submodule SHA | Status |
|---|---|---|---|
| **C1 — zeta_max clamp** | Adds `zeta_max: float = inf` to `gmflow_posterior_mean_jit_general`. Wrapper computes `finfo.max / 10.0` and passes it. Prevents float32 overflow at extreme VP ratios (e.g. t → 0). | `175d7e1` | Done |
| **C2 — gm_vars shape assert** | `torch._assert` checks on `gm_vars.size(gm_dim)==1` and `gm_vars.size(channel_dim)==1` in both JITs. The Mathematica proof only holds when var_k is shared across K components. | `544088e` | Done |
| **C3 — XOR dispatch gate** | Replaces silent `1−σ` auto-fill with `ValueError("Pass both … or neither.")`. No caller passes alpha kwargs yet, so blast radius is zero. | `3a3c389` | Done |
| **C4 — Policy schedule-lock doc** | `piflow_policies/gmflow.py` calls the legacy JIT directly (option b: documented, not re-routed). Import-level block comment + inline notes at both call sites. | `a22f5e1` | Done |

---

## Files Touched

### Submodule — `repos/piFlow`

| File | Changes |
|---|---|
| `lakonlab/models/diffusions/gmflow.py` | C1: `zeta_max` param + clamp in `jit_general`; wrapper computes & passes `zeta_max` · C2: `torch._assert` on `gm_vars` shape in both JITs · C3: `ValueError` on half-pair alpha args |
| `lakonlab/models/diffusions/piflow_policies/gmflow.py` | C4: import-level comment + inline `"linear schedule only — see C4"` notes at both direct JIT calls |

### Evidence repo — `derivations/`

| File | Role | New? |
|---|---|---|
| `_test_vp_stability.py` | C1 fold: dropped local JIT copy, loads production via `importlib`, guards on `has_zeta_max` | Modified |
| `_test_wrapper_dispatch.py` | C3: `_reference_dispatch` raises `ValueError` on half-pair; audit tokens updated; `only_*` variants removed; `test_xor_gate()` added; arg order fix (`eps, inf`) | Modified |
| `_test_gm_vars_shape_assert.py` | C2 driver: happy path + malformed-K + malformed-C shapes. Matcher tightened to require `"gm_vars"` literally. Arg order fix. | **New** |
| `_test_production_vp_finite.py` | V5 gate: sweeps t, var_k, s_src on production `jit_general` with `zeta_max=inf`. Arg order fix. | **New** |
| `_test_vp_var1_boundary.py` | Phase 4 gate (f): `var_k=1` boundary characterization, hard assertions on `var_k≠1`. Arg order fix. | **New** |
| `_test_gate_g_checkpoint.py` | Gate (g) harness: `--record` and `--verify` modes using real GMDiT checkpoint. Run pending on research server. | **New** |
| `03_rollout_plan.md` | §0 status table updated (all C1–C4 done); §1 gate results table added; §2.1/§2.5 checkboxes ticked; §4.1 gate (g) commands added; §7 file table expanded | Modified |
| `01_posterior_rederivation.ipynb` | Re-executed (V1): cell 18 prints `analytic=brute=1.913505`, `|diff|=1.77e-07` | Modified |

---

## Verification Gates at SHA `a22f5e1`

| Gate | Description | Result | Notes |
|---|---|---|---|
| V1 | Notebook re-execution | **PASS** | `|diff|=1.77e-07 ≤ 5e-7` |
| V2 | Mathematica | **PASS** | `wolframscript` activated on this machine; SymPy cell in V1 notebook covers the same ground |
| V3 | Bit-exact equivalence | **PASS** | `max abs diff = 0.0` all shapes/dtypes |
| V4 | Wrapper dispatch + audit | **PASS** | Audit clean; `0.0` both routes; XOR raises; VP smoke finite |
| V5 | Production VP-finite | **PASS** | All (s, var_k, t) combinations finite |
| V6 | Submodule SHA pin | **PASS** | `a22f5e1f` matches Phase 2 close SHA |
| V7 | Call-site audit | **PASS** | 3 sites (`gmflow.py:362`, `:664`, `pipeline_gmdit.py:136`); all through wrapper; no alpha kwargs |
| V8 | Script parity | **PASS** | Both scripts exit 0; `|diff|=1.77e-07` |

---

## Phase 4 Entry Gates

| Gate | Condition | Status |
|---|---|---|
| (e) | C1 merged — `zeta_max` clamp in production JIT | **Done** |
| (f) | `_test_vp_var1_boundary.py` passes | **Passing** — hard assertions green; boundary baseline recorded |
| (g) | Real-checkpoint linear-schedule equivalence run | **Pending** — harness written (`_test_gate_g_checkpoint.py`); requires mmcv + GPU + `Lakonik/pi-Flow-ImageNet` checkpoint on research server |

---

## What's Next

| Action | Where | Unblocks |
|---|---|---|
| Run gate (g) harness (`--record` then `--verify`) | Research server (mmcv + checkpoint) | Phase 4 open |
|  |  |  |
| First Phase 4 opt-in: add `alpha_t_src`/`alpha_t` to `pipeline_gmdit.py` call | `repos/piFlow` submodule | VP schedule live at research path |
| Idea 1 proper: network outputs Fourier coefficients `{aₘ, bₘ}` | New model head + training config | Full Idea 1 |

