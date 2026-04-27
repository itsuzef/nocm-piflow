# Reviewer: General — Independent Assessment

## Summary

The Phase 1 code change is sound and the `03_go_no_go.md` claims are largely supported by reproducible evidence. The Mathematica completing-the-square argument is **correct, but only conditional on `vk` (gm_vars) being shared across the K mixture components** — a precondition that the code unambiguously enforces today (see `lakonlab/models/architecture/gmflow/gm_output.py:82`, `gmflux2.py:261-263`, `gmqwen.py:160-162`, `toymodels.py:114-116`, all emitting `(bs, 1, 1, 1, 1)` logstds). The bit-exactness claim is genuine: I re-ran `_test_differential_equivalence.py` and confirmed `max abs diff = 0.0` across all five (B,K,C,H,W) configs in float32 + float64. The biggest gap in the doc is that the VP stability test exercises a *clamped* variant that is not the function actually shipped. Recommend **GO on Phase 2**, with one tightening (see Risks).

## Math check

- **Information-form derivation** (`01_posterior_rederivation.ipynb` cells 6, 8, 12) is straightforward Gaussian-update algebra and matches the Mathematica derivation in `posterior_rederivation.nb`.
- **Completing-the-square claim**: with `A = −dk/(2vk)`, `B = μk/vk + ν`, `C0 = −μk²/(2vk)`, the marginal log-normalizer (modulo the `½log(−2A⁻¹)` term the notebook says it drops) is `−B²/(4A) + C0 = μk(ν − ½ζμk)/dk + vk·ν²/(2dk)`. So the truncated `ΔLogW` and the full quadratic form differ by exactly `vk·ν²/(2dk)` as claimed in `02_mathematica_crosscheck.md` and the .nb file (line 234-244 of the .nb).
- **k-independence**: that missing term `vk·ν²/(2dk)` depends on `dk = vk·ζ + 1`. It is k-independent **iff vk is shared across k**. Same logic applies to the dropped `½log(vk/dk)` log-determinant term — also k-independent only under shared `vk`. The Go/No-Go doc states this precondition (Test 4) but a reader who only skims the table may miss that the entire correctness argument leans on it. The code enforces it; a future architecture that emits per-component `logstds` (shape `(bs, K, 1, 1, 1)`) would silently break the formula. Worth a `assert gm_vars.shape[-4] == 1` in the JIT body.
- **Trig limit check** (notebook Section 5): `lim t→0⁺` of the posterior mean returns `xt` as expected; `t→1⁻` is degenerate (s<t violates sampling order) — correctly flagged.
- **Notebook Section 6 has a noisy artifact**: cell 18 outputs an `AssertionError` (`Trig mismatch: 0.21555`) before the C=1 fixture passes. The text explains why (per-channel marginal vs joint posterior weights coupled across channels), but the cell ordering means a casual reader sees a failed assert. Recommend either suppressing the broken multi-channel assert or replacing it with a comment-only sanity print.

## Engineering check

- **Bit-exact equivalence** (`_test_differential_equivalence.py`): re-ran locally, all five configs report `max_abs = 0.0` (not just within epsilon). This is because the shipped general function at `repos/piFlow/lakonlab/models/diffusions/gmflow.py:105-109` deliberately mirrors the legacy operator order (`alpha_over_sigma_t = alpha_t / sigma_t; … aos_t * x_t / sigma_t` — two divides, one multiply) rather than the algebraically-equivalent-but-FP-different `alpha_t * x_t / sigma_t**2`. This matters: I noticed the standalone `~/Documents/piFlow/lakonlab/.../gmflow.py:91` uses the latter order and would NOT be bit-exact. The submodule version is the right one to ship; flag this discrepancy with the user (the user's MEMORY.md says the submodule is canonical, which matches).
- **Wrapper dispatch** (`_test_wrapper_dispatch.py`): re-ran, audit + linear equivalence + VP smoke all pass. The dispatch logic at `gmflow.py:215-223` is the obvious right shape (opt-in iff either alpha kwarg is non-None).
- **Float32 safety**: the shipped general function has **no `zeta_max` clamp**, despite the notebook Section 7 deriving an analytic bound and `_test_vp_stability.py` testing a *modified* function that includes one (line 36 of that file). I ran the actual production function on `t ∈ {1e-3, 1e-5, 1e-10, 1e-20}` with both `var=10` and `var=1e-3 + 5σ-scale means`; in all cases output stayed finite. The reason is graceful: when `var*zeta` overflows to `+inf`, both `out_means` numerator and `denom` blow up symmetrically and the ratio resolves to 0, and the softmax then handles the rest. So the "VP path is float32-safe" claim is empirically true for the shipped code, but **not for the reason the test script implies**. The test artefact is misleading.

## Risks / blind spots

1. **Hidden precondition** on shared `vk`. Add a runtime guard or at minimum a docstring note that says "if you ever change the architecture to emit per-component variance, you must also restore the dropped log-Z terms."
2. **VP stability test mismatch** (above). Either ship the `zeta_max` clamp from `_test_vp_stability.py` into the production function (cheap, defensible), or rewrite the test to exercise the actual shipped code so the "float32 safe" evidence corresponds to what's deployed.
3. **No real-checkpoint validation**. The doc itself flags this, but the recommended next step ("run Test 6 on a real checkpoint") is buried at the end of section "What this means concretely" rather than in the Phase 2 entry criteria.
4. **Sign of `zeta` near `t_src` ≈ `t_t`**. When the caller passes nearly-identical source/target times, `zeta` is small and the importance update is nearly identity — but `denom` can dip below `eps=1e-6` and get clamped, causing a one-sided bias. Worth a regression test at `|t_t − t_src| < 1e-3`.
5. **`piflow_policies/gmflow.py:73,81`** still imports the legacy `gmflow_posterior_mean_jit` directly, bypassing the wrapper. Phase 2 should either route this through `gmflow_posterior_mean()` or document why the policy path is intentionally schedule-locked.

## Rollout opinion

The proposed Phase 2/3/4 sequence is appropriate. Two adjustments:

- **Tighten Phase 2 entry criteria**: add "real-checkpoint linear-schedule equivalence (forward-step only, no sample quality assertion)" before merging. This is cheap (one fixture, one CPU run) and converts the synthetic-tensor evidence into something a reviewer can reproduce against the live model.
- **Move the `zeta_max` clamp into Phase 2**, not Phase 4. The cost is one `clamp` op; the benefit is that the `_test_vp_stability.py` evidence matches shipped behaviour. Otherwise Phase 4 will have to re-litigate stability.
- Phase 3's debug-mode equivalence assertion is nice-to-have but probably overkill given Test 6's bit-exact result already lives in CI. Skip unless someone asks.

## Verdict

**GO on Phase 2** as scoped, with two changes: (a) add the analytic `zeta_max` clamp now so test evidence matches shipped code, (b) add a one-line precondition assert (or docstring note) that `gm_vars` must be shared across the K dim. Phase 1 does not need to be revisited.
