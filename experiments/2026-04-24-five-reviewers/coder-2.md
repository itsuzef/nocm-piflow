# coder-2 (critical code-reviewer lens)

> Relayed verbatim from agent output: the `feature-dev:code-reviewer` subagent had no Write tool in its toolset, so the report is saved here by the dispatcher rather than written by the agent directly. Content is unedited.

## Summary

The derivation is mathematically sound and the bit-exact equivalence test is legitimate. However, there is one significant engineering gap: the VP float32 stability test (evidence layer 7 in the go/no-go) tests a *test-only* function that includes a `zeta_max` clamp — that clamp is absent from the shipped production function `gmflow_posterior_mean_jit_general`. Phase 1 is correctly scoped. Phase 2 (the wrapper dispatch) is also correctly implemented and the dispatch test is thorough. The rollout can proceed, but the missing zeta clamp in the production path needs to be resolved before a real VP schedule is used in production (Phase 4).

---

## Math Check

**ζ and ν derivation.** The Mathematica notebook (`02_mathematica_crosscheck.md`, Section 2) defines:
- `ζ = (αt/σt)² − (αs/σs)²`
- `ν = αt·xt/σt² − αs·xs/σs²`

The production JIT (`gmflow.py` lines 105-109) computes:
```python
alpha_over_sigma_t = alpha_t / sigma_t      # αt/σt
nu = alpha_over_sigma_t * x_t / sigma_t     # (αt/σt)(xt/σt) = αt·xt/σt²
```
This is algebraically identical to the notebook's ν. No discrepancy.

**`vk·ν²/(2·dk)` cancellation.** The go/no-go claims this term is k-independent under softmax (evidence row 2). The argument: `dk = vk·ζ + 1` and `vk` are both shared across the K components (gm_vars has shape `(bs,*,1,1,1,1)` — the K axis is size 1). Therefore `vk·ν²/(2·dk)` is identical for all k and cancels under softmax. This is correct. The code-path audit confirming `gm_vars` is never per-component (evidence row 4) is the load-bearing precondition and appears valid from reading `gmflow.py` lines 205-209.

**Brute-force test.** `_run_numerical.py` lines 128-139 compute the correct Bayesian ratio `log p(xt|x0) − log p(xs|x0)` and add it to the GM log-prior. This is a genuine independent check of the multi-component update, not circular. The 1D, K=4 comparison at `_run_numerical.py` line 171 with tolerance `< 0.01` (loose but appropriate for grid quadrature) is valid.

---

## Engineering Check

**Bit-exactness claim.** The claim that the general function is bit-exact with the legacy function on the linear-schedule path is plausible and well-tested. Both functions use the same two-division operator ordering for `nu`: `(alpha/sigma) * x / sigma`, which matches exactly because the legacy function also computes `alpha_over_sigma_t * x_t / sigma_t` (`gmflow.py` line 58). The test (`_test_differential_equivalence.py`) extracts both functions directly from the production source via regex+importlib, so it is testing the actual deployed code, not a transcription.

**Critical gap — stability test does not test the production function.** `_test_vp_stability.py` defines and tests `gmflow_posterior_mean_general` (lines 19-47 of that file) — a local function with a `zeta_max` clamp on line 36. The production function `gmflow_posterior_mean_jit_general` (`gmflow.py` lines 75-123) has NO zeta clamp. The go/no-go document (evidence row 7) states "All outputs finite and bounded" but this refers to the test-only variant. If a caller passes a genuine VP schedule with very small `t` (e.g. `t = 1e-5`), `zeta = (cos(πt/2)/sin(πt/2))² ≈ (2/(πt))²` grows as ~4e9 at `t=1e-5`. With `gm_vars` even moderately large, `gm_vars * zeta` overflows float32. The production function will produce `inf` in that case; the test proved it would not only because the clamp was present.

**Wrapper dispatch.** Phase 2 is correctly implemented. The dispatch logic at `gmflow.py` lines 211-223 is clean: `use_general = alpha_t_src is not None or alpha_t is not None`. The `_test_wrapper_dispatch.py` source-audits the exact dispatch tokens and tests all four opt-in patterns. No issues found.

**`gm_vars` cache side-effect.** `gmflow_posterior_mean` at line 209 writes `gm['gm_vars'] = gm_vars` into the caller's dict on every invocation when `logstds` is used. This is a pre-existing side effect (not introduced by the refactor), but it means callers that reuse the `gm` dict across calls may see unexpected mutation. Not a new bug, but worth noting.

---

## Risks and Blind Spots

1. **Production VP path is not float32-safe without the zeta clamp.** The stability test's `ZETA_MAX` clamp lives only in `_test_vp_stability.py` and is not in the shipped function. Any Phase 4 experiment that routes a VP schedule through the production function at very small `t` will overflow silently (outputs become `inf` which, after softmax, may collapse to `nan` or a degenerate distribution). This is the highest-severity gap.

2. **Negative zeta path.** If a caller mistakenly passes `t > t_src` (querying a noisier time than the source), `zeta < 0`. `denom = gm_vars * zeta + 1` can go negative for large `gm_vars`, and the `.clamp(min=eps)` will silently snap it to eps rather than raising. The output will be numerically garbage with no warning. No test exercises this. The go/no-go does not mention it.

3. **Multi-channel weight update assumption.** The logweights_delta sums over the channel dimension (`dim=channel_dim`) and divides by `denom` (which is scalar across channels since `gm_vars` has channel dim 1). This is correct only if the observation noise is isotropic (shared `gm_vars` across channels). For the current GMFlow this holds, but if someone uses a per-channel `gm_vars` in the future, the formula silently produces wrong posteriors.

4. **No test on real checkpoint activations.** As the go/no-go itself acknowledges (page 2), synthetic Gaussian `randn` tensors may miss pathological real-model outputs (very confident weights, extreme means). The stability tests give no signal for that regime.

5. **`_run_numerical.py` brute-force tolerance.** The 1D brute-force comparison uses `diff < 0.01` as PASS (`_run_numerical.py` line 173). This is 10 million times looser than the `1e-7` claimed in the go/no-go table for "machine epsilon." The 1.79e-07 figure is the max diff on the linear-schedule path (cell 18 of the notebook), not for the trig-schedule brute-force. The go/no-go's evidence table conflates these two results in row 3.

---

## Rollout Opinion

Phase 1 and Phase 2 are sound; shipping them is low risk. Phase 3 (debug equivalence assertion) is a good idea and does not require the zeta clamp to be resolved first — it only fires on the linear-schedule path. Phase 4 should be blocked until the production function receives the `zeta_max` clamp (a two-line change already demonstrated in `_test_vp_stability.py`), so that VP experiments at small `t` cannot silently overflow. The clamp should be added to `gmflow_posterior_mean_jit_general` itself, not left as test-only infrastructure.

Recommended sequencing: (1) Add zeta clamp to production function. (2) Rerun `_test_vp_stability.py` against the production function (not the test-only variant) to confirm. (3) Then proceed with Phase 4.

---

## Verdict

**Hold Phase 4; ship Phase 1–3.** Phase 1 and 2 are correctly implemented and well-tested. The critical risk is that the float32 stability evidence for the VP schedule path tests a test-only function variant, not the shipped production function. Before any real VP schedule is activated in a sampler, the `zeta_max` clamp must be ported into `gmflow_posterior_mean_jit_general`. This is a small, targeted fix but it is a pre-condition for VP safety, not something to discover during a real inference run.
