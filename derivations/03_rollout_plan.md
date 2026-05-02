# GMFlow Schedule-Agnostic Posterior — Technical Reference

**Subject**: Generalising `gmflow_posterior_mean_jit` from the hardcoded linear
schedule (`α = 1 − σ`) to an arbitrary `(α, σ)` pair, then opting call sites
in one at a time.

**Companion docs**: `01_posterior_rederivation.ipynb`, `02_mathematica_crosscheck.md`,
`posterior_rederivation.nb`.

---

## Status overview

| Item | State |
|---|---|
| `gmflow_posterior_mean_jit_general` exists alongside the legacy JIT | Shipped |
| Wrapper `GMFlowMixin.gmflow_posterior_mean` accepts optional `alpha_t_src` / `alpha_t` and dispatches | Shipped; hygiene items complete |
| `gm_vars` shape assertion inside the general JIT | Done — both JITs assert `size(gm_dim)==1` and `size(channel_dim)==1` |
| XOR-gate on the wrapper's `(alpha_t_src, alpha_t)` pair | Done — passing exactly one raises `ValueError` |
| `piflow_policies/gmflow.py` | Intentionally schedule-locked (linear only) on three surfaces — see "VP completeness" below |
| `zeta_max` clamp in production | Done — `zeta_max: float = inf` param added; wrapper computes and passes `finfo.max / 10.0` |
| Bit-exact equivalence (`max abs diff = 0.0`, 20 seeds × 5 configs × fp32+fp64) | Verified |
| Production VP path stays finite at small `t` | Verified |

---

## Verification results

Eight gates were run at the current codebase revision.

| Gate | Description | Result | Notes |
|---|---|---|---|
| V1 | Notebook re-execution | **PASS** | Exit 0; `\|diff\|=1.77e-07 ≤ 5e-7` |
| V2 | Mathematica cross-check | **PASS** | SymPy (cell 7 in V1 notebook) confirms `missingTerm=0`, `μk ∉ free_symbols` |
| V3 | Bit-exact equivalence | **PASS** | `max abs diff = 0.0` all shapes/dtypes |
| V4 | Wrapper dispatch + source audit | **PASS** | Audit clean; both routes `0.0`; XOR gate raises; VP smoke finite |
| V5 | Production VP-finite | **PASS** | All (s, var_k, t) combinations finite |
| V6 | Codebase SHA pin | **PASS** | Confirmed at close |
| V7 | Call-site audit | **PASS** | 3 call sites (all routing through wrapper; no alpha kwargs) |
| V8 | Script parity | **PASS** | Both scripts exit 0; trig block `\|diff\|=1.77e-07 ≤ 5e-7` |

---

## Completed work

### C1 — zeta_max clamp

Adds a `zeta_max: float` argument to `gmflow_posterior_mean_jit_general` (default `inf`; wrapper computes and passes `finfo.max / 10.0`). Prevents float32 overflow at extreme VP ratios (e.g. t → 0). `_test_vp_stability.py` now exercises the production function directly.

### C2 — gm_vars shape assertion

`torch._assert` checks on `gm_vars.size(gm_dim)==1` and `gm_vars.size(channel_dim)==1` inside both JITs. The derivation only holds when variance is shared across K mixture components. `_test_gm_vars_shape_assert.py` covers the happy path and malformed shapes.

### C3 — XOR dispatch gate

Replaces the silent `1 − sigma_*` auto-fill with an explicit `ValueError` when exactly one of `(alpha_t_src, alpha_t)` is `None`. No existing call site passes either alpha kwarg, so this is zero-blast-radius. `_test_wrapper_dispatch.py` updated to assert the raise.

### C4 — Policy schedule-lock documentation

`piflow_policies/gmflow.py` calls the legacy JIT directly (schedule-locked by design, for performance reasons). Import-level block comment and inline notes document this intent at both call sites.

---

## Call-site inventory

| File | Lines | Routing |
|---|---|---|
| `lakonlab/models/diffusions/gmflow.py` | Wrapper dispatch internals | Routes to general or legacy path |
| `lakonlab/models/diffusions/gmflow.py` | `gm_2nd_order` | Routes through wrapper |
| `lakonlab/pipelines/pipeline_gmdit.py` | Pipeline-level call | Routes through wrapper |
| `lakonlab/models/diffusions/piflow_policies/gmflow.py` | Two direct JIT calls | Schedule-locked (C4); documented |

---

## VP completeness — outstanding surface area

The schedule-agnostic posterior is one slice of what an end-to-end VP-capable
pipeline needs. A separate audit traced the `u → x_0` conversion and adjacent
variance-rescaling sites and confirmed that **the rest of the codebase is
still linear-schedule-only**. None of these are live bugs today (the
`(alpha_t_src, alpha_t)` kwargs added to the wrapper do not propagate beyond
it, and no in-tree config selects a non-linear sampler), but every site below
becomes a silent miscompute the moment a non-linear `(α, σ)` reaches it.

### Schedule-locked surfaces

| Surface | Locations | Form today | Form needed under VP |
|---|---|---|---|
| `u → x_0` mean | `GaussianFlow.u_to_x_0`, `GMFlowMixin.u_to_x_0` (3 branches), `DXPolicy._u_to_x_0`, `GMFlowPolicy._u_to_x_0` (4 implementations total) | `x_0 = x_t − σ · u` — exact only when `α = 1 − σ` | needs an `α` factor; correct form to be derived alongside Phase 4 |
| `u → x_0` variance / log-std | `GMFlowMixin.u_to_x_0` log-std branch (`logstds_x_0 = logstds + log σ`); `GMFlowPolicy._u_to_x_0` (`gm_vars = exp(2·logstds) · σ²`) | `σ²` rescaling | likely involves `(σ/α)²` or similar; needs re-derivation, not assumed |
| Forward training process | `GaussianFlow.sample_forward_diffusion` | hardcoded `mean = 1 − σ` | requires a schedule object passed into the training loop |
| Policy classes | `DXPolicy`, `GMFlowPolicy` (`BasePolicy.pi(x_t, σ_t) → u` has no `α`) | `(x_t, σ_t)`-only contract | `α`-aware contract or schedule-bound policy construction |

### Trigger conditions for the latent bugs

| Trigger | Sites lit | How it happens |
|---|---|---|
| **A** — test-time scheduler swap | `GaussianFlow.forward_test/forward_u`, `GMFlow.forward_test/forward_u`, `GMDiTPipeline.__call__` | A user picks a non-flow `diffusers` sampler via `test_cfg_override.sampler` (or constructs the pipeline with one). Today no in-tree config does this; not enforced at runtime. |
| **B** — VP rollout into the policy classes | `GMFlowMixin.gmflow_posterior_mean` (`prediction_type='u'` branch); `pipeline_gmdit`; `DXPolicy`; `GMFlowPolicy` | Phase 4 opts a call site into VP. Sites #7–#9 of the audit go live. |

### What Phase 4 must address

Before any call site flips to a non-linear schedule:

1. Decide each linear-locked surface's correct VP form (re-derive `u → x_0` mean and variance under arbitrary `(α, σ)`; do not assume the `σ²` form generalises).
2. Either fix all four `u_to_x_0` implementations or guard them at the API boundary so a non-linear `(α, σ)` cannot reach them silently.
3. Decide whether `BasePolicy.pi(x_t, σ_t) → u` gains an `α` argument or is replaced by an `α`-aware policy construction step.
4. Add a runtime guard against Trigger A (e.g., assert `α + σ ≈ 1` inside `u_to_x_0` until the form is generalised).

The current `gmflow_posterior_mean_jit_general` work is necessary for any of
this but does not on its own make the pipeline VP-correct.

---

## Remaining gate

Phase 4 (opting live call sites in to the VP schedule) is blocked on one remaining gate:

**Gate (g)** — Real-checkpoint linear-schedule equivalence run.

- Harness written (`_test_gate_g_checkpoint.py`).
- Requires GPU + checkpoint access (mmcv/mmgen environment).
- Procedure: record sampling outputs at the pre-change revision; verify bit-exact match at the current revision.
- Until this passes, the legacy linear-schedule path remains default; no behaviour change is live.

Once gate (g) closes, rollout proceeds one call site at a time, starting with the `pipeline_gmdit` research path, then the second-order corrector, and finally the policy (if the policy is ever re-routed through the wrapper).

---

## Files in this evidence bundle

| Path | Role |
|---|---|
| `01_posterior_rederivation.ipynb` | SymPy + numerical derivation. Cell 17 is the load-bearing trig-schedule check (V1). |
| `posterior_rederivation.nb` | Mathematica completing-the-square proof. Re-runnable. |
| `02_mathematica_crosscheck.md` | Human-readable Mathematica companion + verified-output appendix. |
| `_replay_nb.wls` | `wolframscript -file` driver for Mathematica verification. |
| `_run_sympy.py` | Script form of the SymPy derivation (V8). |
| `_run_numerical.py` | Script form of the numerical tests (V8). |
| `_test_differential_equivalence.py` | Bit-exactness sweep (V3). |
| `_test_vp_stability.py` | VP stability test against production function. |
| `_test_wrapper_dispatch.py` | Dispatch and XOR gate verification (V4). |
| `_test_production_vp_finite.py` | Production VP-finite sweep (V5). |
| `_test_gm_vars_shape_assert.py` | Shape assertion test (C2). |
| `_test_vp_var1_boundary.py` | `var_k=1` boundary characterization; hard assertions pass. |
| `_test_gate_g_checkpoint.py` | Gate (g) harness; run pending. |
| `03_rollout_plan.md` | This document. |
