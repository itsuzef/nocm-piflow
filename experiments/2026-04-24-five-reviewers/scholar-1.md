# scholar-1 — Independent Review of GMFlow Posterior Schedule-Agnostic Refactor

## Summary

The derivation is correct, the bit-exactness claim is sound, and Phase 1 (sibling JIT) is genuinely zero-blast-radius. I independently re-derived the completing-the-square argument in SymPy and reproduced the missing-term identity `(missing) = vk·ν²/(2·dk)` with no `µ_k` dependence — so it cancels under softmax exactly as claimed, *provided* `vk` is shared across mixture components. That precondition holds throughout the GMFlow code paths I audited. Phase 2 ships a wrapper whose default branch dispatches to the legacy JIT verbatim, so no caller sees a behavior change. I recommend **GO on Phase 2**, with two small documentation/guard tightenings before Phase 4.

## Math check

The Mathematica notebook (`derivations/posterior_rederivation.nb:135-243`) constructs `logQ = -(x0-µk)²/(2vk) - (ζ/2)x0² + ν·x0`, reads off `A,B,C0`, then forms `logZ_full = -B²/(4A) + C0` and asks Mathematica to verify `logZ_full - ΔLogW == vk·v²/(2·dk)`. The structure of that argument is exactly the standard Gaussian × Gaussian-form-correction completing-the-square (drop the `½ log(-2A)` normalizer because it's k-independent; what's left is a quadratic form in `(B, A)`).

I re-ran the same algebra in SymPy from scratch (no shared code with the verification scripts). Result: `simplify(logZ_full - DeltaLogW - vk*nu**2/(2*dk)) == 0` and `mu_k not in missing.free_symbols`. So the symbolic identity is real, not a Mathematica artifact.

The crucial *physical* assumption powering "drops under softmax" is **`vk` shared across `k`**. The notebook flags this; `02_mathematica_crosscheck.md:9` lists `vk > 0` as a single scalar. Code-side, `repos/piFlow/lakonlab/models/diffusions/gmflow.py:208` builds `gm_vars` from `gm['logstds']` whose canonical shape (per `lakonlab/ops/gmflow_ops/gmflow_ops.py:72,82,110,162`) is `(bs, *, 1, 1, 1, 1)` — broadcast over the K-axis. So the precondition holds in the live code, not just in theory. Good.

The SymPy notebook (Test 3, `01_posterior_rederivation.ipynb` cell 18, |diff|=1.79e-7 vs 1D brute force on a K=4 trig schedule) is a meaningful independent check that the *formula* (not just the symbolic identity) yields the correct posterior mean of the importance-reweighted GM, because the brute force grid integrates the true `p(x0|x_t,x_s)` directly.

One minor pedagogical gap: `02_mathematica_crosscheck.md` doesn't explicitly state why "k-independent term drops under softmax" — it's standard but worth one sentence for readers who aren't fluent. Not a math error.

## Engineering check

The bit-exactness claim is plausible and I'd accept it. The two JIT functions in `gmflow.py:43-72` and `gmflow.py:75-123` use *literally the same float ops in the same order* once `alpha_t = 1 - sigma_t` is supplied externally. Critical operator-order detail: line 58 vs line 109 both compute `aos * x / sigma`, two divides — not `aos * x / (sigma_t**2)`. IEEE-754 is deterministic for identical op trees on identical inputs, so `max abs diff = 0.0` across 20 seeds × 5 configs (Test 6) is exactly what I'd predict, not something to be surprised by. Good engineering instinct in the notebook comment at `gmflow.py:98-100`.

One small wart: `derivations/_test_vp_stability.py:19-47` defines its **own** local copy of the JIT with a `zeta_max` clamp that the **production** `gmflow_posterior_mean_jit_general` does not have (`gmflow.py:108` clamps neither sign of zeta). So Test 7 validates a stability strategy, not the production code. The `_run_numerical.py` Section 7 derivation of `ZETA_MAX_safe = float32_max / max_var_k` is correct, but it's not enforced in the shipped function. For VP at very small `t` you currently rely on the user clamping `sigma_t >= eps` and on `(gm_vars * zeta + 1).clamp(min=eps)` to absorb the overflow — which works for `denom` but `out_means = (gm_vars * nu + gm_means) / denom` could still overflow `nu`. Not a Phase 1/2 blocker; flag for Phase 4.

## Risks / blind spots

1. **Wrapper auto-fill silently mixes schedules.** `gmflow.py:213-216`: if a caller passes only `alpha_t_src` (e.g. from VP), `alpha_t` defaults to `1 - sigma_t` (linear). That's mathematically a different schedule — the math is still correct, but the *semantics* are wrong and there's no warning. I'd require both or neither.
2. **`gm_vars` shape invariant is asserted nowhere in JIT.** The k-independence proof rests entirely on `gm_vars` broadcasting over the K-axis. If a future contributor introduces per-component variances, the truncated `ΔLogW` becomes wrong without any test failing. A `torch._assert(gm_vars.size(gm_dim) == 1)` inside the JIT is cheap insurance.
3. **Synthetic-tensor blind spot already flagged but not retired.** `03_go_no_go.md:35` correctly notes real-checkpoint validation is open. Phase 4 ships a *schedule change*, not just a refactor — ship-quality bar should include at least one real-checkpoint run with `alpha = 1 - sigma` to confirm wrapper plumbing, then a separate experiment with VP.
4. **Float32 overflow at small `t` in VP**: see Engineering check above. The production JIT lacks the `zeta_max` clamp from `_test_vp_stability.py`.
5. **Test 6 covers only `alpha = 1 - sigma`.** That's the right choice for "no-op" certification, but it doesn't prove the *general* function is correct on VP at full image shape — only the 1D brute-force in Test 3 does, at C=1, K=4. Adding a multi-channel brute-force (or a Monte-Carlo posterior estimator) would be cheap.

## Rollout opinion

The Phase 1→4 sequencing in `03_go_no_go.md:39-100` is sound. I'd amend:

- **Phase 2 (now)**: ship the wrapper, but tighten the dispatch gate at `gmflow.py:211` to "both alphas or neither" — XOR is a foot-gun. Also add the `gm_vars` shape assertion inside `gmflow_posterior_mean_jit_general`.
- **Phase 3**: the proposed env-gated allclose check is fine. Make sure it compares to `gmflow_posterior_mean_jit` (legacy), not to itself.
- **Phase 3.5 (new)**: real-checkpoint linear-schedule run, asserting bit-exactness vs the previous git SHA. Cheap, decisive.
- **Phase 4**: introduce the analytic `zeta_max` clamp from `_run_numerical.py:183` into the production general JIT *before* enabling VP at any call site. Then opt in one call site at a time.

## Verdict

**Ship Phase 2 now**, with the XOR-dispatch tightening and the `gm_vars` shape assertion added in the same PR. Hold Phase 4 until (a) real-checkpoint validation lands (Phase 3.5) and (b) the analytic zeta clamp is moved into the production general JIT. Phase 1 does **not** need to be revisited — the math is correct and the bit-exact equivalence is honest.
