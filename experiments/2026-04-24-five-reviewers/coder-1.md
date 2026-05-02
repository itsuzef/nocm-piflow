# coder-1 (code-explorer lens)

> Relayed verbatim from agent output: the `feature-dev:code-explorer` subagent had no Write tool in its toolset, so the report is saved here by the dispatcher rather than written by the agent directly. Content is unedited.

## Summary

Phase 1 (sibling JIT function) is correctly engineered for the linear-schedule path: the algebra is sound, and the bit-exactness claim on the existing path is genuine. However, the go/no-go document contains a material misrepresentation: it cites `1.79e-07` for "numerical brute-force vs analytic on C=1, K=4 trig schedule" (Test 3), but the actual notebook (`01_posterior_rederivation.ipynb`, cell 18) shows a hard `AssertionError: Trig mismatch: 0.21555089950561523` for that exact test. The VP/trig path has not been numerically validated. Additionally, the shipped `gmflow_posterior_mean_jit_general` contains no `zeta_max` clamp, while the stability test (`_test_vp_stability.py`) tests a completely different function that does. Phase 2 should be held until the trig discrepancy is resolved and the stability test is updated to test the actual shipped function.

---

## Math Check

The completing-the-square argument for dropping `vk*nu^2/(2*dk)` under softmax is correct. In the Mathematica notebook (`02_mathematica_crosscheck.md`, Section 2), `ΔLogW = mu_k*(nu - (1/2)*zeta*mu_k)/dk`. The term `vk*nu^2/(2*dk)` is indeed k-independent when `var_k` (= `vk`) is shared across all K components — it appears identically in every component's log-weight and cancels under softmax. The Mathematica and SymPy derivations agree; linear-schedule reduction to the existing code is verified symbolically (`diff_zeta = 0`, `diff_nu = 0`; notebook cells 12 and 14 in `02_mathematica_crosscheck.md` / `01_posterior_rederivation.ipynb`).

The information-form derivation (`repos/piFlow/lakonlab/models/diffusions/gmflow.py:44–72`) is internally consistent. The `nu` operator order (`(alpha/sigma) * x / sigma`) used in production is algebraically identical to the canonical `alpha*x/sigma^2`.

One blind spot: the Mathematica notebook (Section 5) computes `lim_{t->1} mu_VP = (-mu_k + sqrt(2)*var_k*x_s)/(var_k - 1)` (`01_posterior_rederivation.ipynb`, cell 15 output). When `var_k = 1`, this diverges. The derivation does not flag this as a degenerate case requiring attention in the VP sampler.

---

## Engineering Check

**Bit-exactness (linear path)**: Genuine. `_test_differential_equivalence.py` extracts both functions directly from the production source file, hashes them, and runs 20 seeds × 5 shape configs in float32 + float64. The test verifies `max_abs = 0.0` at all configs. The operator-order comment in `gmflow_posterior_mean_jit_general` (`gmflow.py:98-100`) correctly explains why: `nu` is computed as `aos * x / sigma` (two divides), mirroring the legacy code exactly.

**Trig/VP path — critical defect**: Notebook cell 18 (`01_posterior_rederivation.ipynb`) runs a C=1, K=4 brute-force comparison for the trig schedule and fails with `AssertionError: Trig mismatch: 0.21555089950561523` (analytic=1.697954, brute=1.913505). The go/no-go document (Test 3 row, `03_go_no_go.md:19`) claims "`|diff| = 1.79e-07` (machine epsilon)" for this test. That number does not appear anywhere in the notebook output. The `1.79e-07` figure is from the linear-schedule cross-check (notebook cell 17 output: `1.19e-07`; close, plausibly from a different random seed). The trig verification was either not run when the go/no-go was written, or was incorrectly attributed.

**Stability test tests wrong function**: `_test_vp_stability.py:19-47` defines and tests `gmflow_posterior_mean_general` — a local function with a `zeta_max` parameter. The shipped production function `gmflow_posterior_mean_jit_general` (`gmflow.py:75-123`) has no such clamp. The stability test does not import or test the production function at all. A VP run on the production code near t=0 will be unguarded.

**GMFlowPolicy not updated**: `piflow_policies/gmflow.py:7` imports `gmflow_posterior_mean_jit` directly and uses it at lines 73 and 81. This is a fifth call site the go/no-go lists (`03_go_no_go.md`, Test 5 row) as "linear-schedule, backward-compatible." Phase 2 does not update this caller. If the policy is ever used with a VP schedule, it will silently use the hardcoded linear formula.

**Wrapper routing logic**: `gmflow.py:211` gates on `alpha_t_src is not None or alpha_t is not None`. If only one of the pair is provided, the other is filled with `1 - sigma_*`. This is sensible but creates a half-VP mode (one end VP, one end linear) that is never tested and could produce wrong posteriors if a caller provides only `alpha_t` by accident.

---

## Risks / Blind Spots

1. **Failed trig verification, not acknowledged**: The go/no-go cites a passing result that is contradicted by the actual notebook output. This is the most serious issue. The formula may be correct and the discrepancy may be an artifact of how the brute-force grid is constructed — but this has not been investigated, and the claim of `1.79e-07` is not supported by the evidence file.

2. **No zeta_max clamp in production function**: For VP near t=0, `zeta` grows as `4/(pi^2 * t^2)`. At t=1e-3 with var_k=1, `zeta*var_k ≈ 4e6`, within float32 range. At t=1e-5, it overflows. The `_test_vp_stability.py` only tests a version with the clamp; the shipped function is unprotected.

3. **var_k=1 degeneracy**: As noted above, the VP boundary limit diverges when `var_k = 1` exactly. If a model outputs logstds such that `exp(2*logstds) = 1`, the VP posterior is degenerate. No test covers this.

4. **Synthetic-only equivalence test**: `_test_differential_equivalence.py` uses `torch.randn` for `gm_means`. Real checkpoint outputs may have extreme values (very confident components, near-zero weights) that exercise different floating-point paths.

5. **Phase 3 debug assertion incompatible with JIT**: The proposed Phase 3 debug assertion (call legacy function and assert allclose when `alpha ≈ 1 - sigma`) cannot be implemented inside a `@torch.jit.script` function. JIT does not support Python-level env-var reads or conditional imports. Phase 3 would require a Python-level wrapper, not a JIT modification.

---

## Rollout Opinion

Phase 1 is safe as shipped: it is a no-op sibling addition. Phase 2 (wrapper) is architecturally correct but should not ship until:

1. The trig mismatch in notebook cell 18 is resolved — either by finding the bug in the brute-force setup, or by fixing the analytic formula.
2. `gmflow_posterior_mean_jit_general` is patched to include a `zeta_max` clamp, and `_test_vp_stability.py` is updated to import and test the actual production function.
3. The go/no-go evidence table is corrected to reflect actual notebook output.

Phase 4 (non-linear schedule in a live call site) should be blocked until Phase 2 issues are resolved and at least one real-checkpoint VP run produces visually plausible samples.

The proposed sequencing (Phases 3 before 4) is reasonable but Phase 3 as described cannot be implemented in JIT. Recommend dropping Phase 3 or moving the assertion to the Python wrapper level outside JIT.

---

## Verdict

**Hold Phase 2.** Phase 1 is safe and can remain. The trig/VP verification layer failed and this failure was misrepresented in the go/no-go document. The production function is missing the zeta clamp that the stability test implicitly requires. Neither issue is blocking for the existing linear-schedule callers, but both must be resolved before any VP-schedule experiment can be trusted.
