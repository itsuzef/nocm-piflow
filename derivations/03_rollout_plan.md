# GMFlow Posterior Schedule-Agnostic Refactor — Go/No-Go Decision

**Subject**: Generalising `gmflow_posterior_mean_jit` from hardcoded linear
schedule (`α = 1 − σ`) to an arbitrary `(α, σ)` pair.

**Original date**: 2026-04-23
**Last revised**: 2026-04-26 — see [Revision history](#revision-history) at the bottom.

---

## Verdict: **GO Phase 1**, **Conditional GO Phase 2**, **HOLD Phase 4**.

Phase 1 (sibling JIT) shipped clean: math correct, bit-exact equivalence
honest, zero blast radius on the linear-schedule path. Phase 2 (wrapper) is
ready to ship **bundled with four small additions** (see Phase 2 below). Phase
3 (debug allclose) is demoted: the originally-proposed JIT-internal version is
not implementable; the bit-exact CI test largely covers its purpose. Phase 4
(VP at a live call site) is blocked on three explicit gates.

Five independent reviewer reports + a parallel cleanup pass converged on the
same picture — see `experiments/2026-04-24-five-reviewers/SYNTHESIS.md`.

---

## Evidence

| # | Test | Result |
|---|---|---|
| 1 | SymPy symbolic derivation of `ζ`, `ν`, `μ_post`, `Δlog w` | All formulas correct. See `01_posterior_rederivation.ipynb` cells 04, 09, 11. |
| 2 | Completing-the-square: full log-normaliser minus truncated `Δlog w` equals `vk·ν²/(2·dk)`, which is **k-independent** when `var_k` is shared across components | Two CAS engines agree. SymPy: `01_posterior_rederivation.ipynb` cell 06 (`mu_k in free_symbols == False`). Mathematica: `posterior_rederivation.nb` (verified output in `02_mathematica_crosscheck.md` appendix). |
| 3 | Numerical brute-force vs analytic on C=1, K=4 trig (VP) schedule | `analytic = 1.913505`, `brute = 1.913505`, `\|diff\| = 1.77e-07`. See `01_posterior_rederivation.ipynb` cell 17 ("Test 2") and `_run_numerical.py` Section 6c. |
| 4 | `gm_vars` shape invariant audit across `gmflow.py`, `gmflow_ops.py`, `piflow_policies/gmflow.py` | Var is shared across components in every code path (shape `(bs, *, 1, 1, 1, 1)`). Mathematica's k-independence precondition holds in production. |
| 5 | Call site audit | 5 call sites, all linear-schedule. Backward-compatible API expansion is straightforward. **Caveat**: `piflow_policies/gmflow.py:73,81` imports `gmflow_posterior_mean_jit` directly and bypasses the wrapper — see Known caveats. |
| 6 | **Bit-exact equivalence** test on full `(B, K, C, H, W)` synthetic data, 20 seeds × 5 shape configs, float32 + float64 | **`max abs diff = 0.0`** at every config (`_test_differential_equivalence.py`). The two JIT functions share an identical operator order on `aos·x/σ`, so IEEE-754 determinism predicts this exactly. |
| 7 | Float32 stability sweep under VP schedule, `t ∈ [1e-20, 0.85]` | All outputs finite (`_test_vp_stability.py`). **Caveat**: this test exercises a *test-only* function with a `zeta_max` clamp that is **not present** in shipped `gmflow_posterior_mean_jit_general`. See Known caveats #1. |

---

## Known caveats

These are real but scoped. Items marked **[blocking]** must close before the
phase they block; items marked **[hygiene]** should ship in the same PR but
are not strict pre-conditions.

1. **Production `gmflow_posterior_mean_jit_general` has no `zeta_max` clamp** — the
   stability evidence in Test 7 is for a non-production variant. Empirically
   the production function still produces finite output at small `t` because
   `inf/inf` cancels symmetrically in `out_means / denom`, but that is fragile
   and unintentional. **[blocking Phase 4]** — port the analytic clamp from
   `_test_vp_stability.py:36` into the shipped JIT.

2. **`gm_vars` shape invariant is not asserted in JIT** — the entire
   correctness argument (Test 2's k-independence) depends on `gm_vars` being
   shared across the K dim. A future architecture emitting per-component
   variances would silently break the formula with no test failing.
   **[hygiene, Phase 2 PR]** — add `torch._assert(gm_vars.size(gm_dim) == 1)`
   inside the general JIT.

3. **Wrapper auto-fill silently mixes schedules** — `gmflow.py:213-216`: if a
   caller passes only one of `(alpha_t, alpha_t_src)`, the other defaults to
   `1 − sigma_*` (linear). Mathematically still correct, but semantically
   wrong, with no warning. **[hygiene, Phase 2 PR]** — XOR-gate the dispatch:
   require both alphas or neither.

4. **`piflow_policies/gmflow.py:73,81`** imports `gmflow_posterior_mean_jit`
   directly and bypasses the wrapper. Phase 2 leaves this call site on the
   legacy code path. **[hygiene, Phase 2 PR]** — route through the wrapper or
   document why it's intentionally schedule-locked.

5. **VP boundary divergence at `var_k = 1`** — the analytic `t → 1⁻` limit
   under VP equals `(√2·var_k·x_s − μ_k)/(var_k − 1)`, which diverges when
   `var_k = 1` exactly. Confirmed independently by Mathematica and SymPy. The
   shipped JIT has no guard. **[blocking Phase 4]** — add a regression test
   exercising `var_k ≈ 1` near `t = 1`.

6. **Synthetic-tensor blind spot** — every numerical test in this evidence
   bundle uses `torch.randn` for GM means/weights. Real checkpoint outputs
   may have extreme means or near-degenerate weights that exercise different
   floating-point paths. **[blocking Phase 4]** — one real-checkpoint
   linear-schedule run, asserting bit-exactness vs the previous git SHA.

7. **Phase 3 debug assertion is not implementable as originally specified** —
   the proposed `if alpha ≈ 1 - sigma: assert allclose(legacy)` cannot live
   inside `@torch.jit.script` (no env-var reads, no Python-level conditional
   imports). **[Phase 3 design change]** — either drop Phase 3 (Test 6's
   bit-exact CI gate already covers it), or move the assertion to a
   Python-level wrapper outside the JIT.

---

## Recommended rollout

### Phase 1 — Land sibling function (shipped, zero blast radius)

`gmflow_posterior_mean_jit_general` lives next to `gmflow_posterior_mean_jit`
in `repos/piFlow/lakonlab/models/diffusions/gmflow.py`. The legacy function is
untouched. No caller invokes the new function yet.

### Phase 2 — Wrapper dispatch (ready to ship; bundle the four hygiene items)

Update `gmflow_posterior_mean()` to accept optional `alpha_t_src` / `alpha_t`
and dispatch to the general JIT when either is provided. Same PR includes:

- **(a)** XOR-gate the wrapper dispatch (caveat #3).
- **(b)** `gm_vars` shape assertion inside the general JIT (caveat #2).
- **(c)** Route `piflow_policies/gmflow.py:73,81` through the wrapper (caveat #4).
- **(d)** Port the analytic `zeta_max` clamp from `_test_vp_stability.py:36`
  into `gmflow_posterior_mean_jit_general` (caveat #1).

The 5 existing call sites that don't pass `alpha_*` are bit-exact-unchanged in
behaviour (Test 6 verifies).

### Phase 3 — Debug-mode assertion (deferred / redesigned)

Original design (JIT-internal allclose vs legacy) is not implementable. Two
options:

- **Drop**: Test 6's bit-exact CI gate already provides the equivalence check.
  Recommended unless someone hits a divergence in practice.
- **Redesign**: a Python-level wrapper that, when `os.environ['GMFLOW_DEBUG']`
  is set, calls both functions and asserts. Pure Python, lives outside any
  `@torch.jit.script` boundary.

### Phase 4 — Schedule-aware sampler experiments (BLOCKED)

Blocked on **all three** of:

- **(e)** `zeta_max` clamp ported into production (caveat #1).
- **(f)** `var_k = 1` regression test (caveat #5).
- **(g)** Real-checkpoint linear-schedule equivalence run (caveat #6).

Once unblocked, opt in one call site at a time (likely the research path in
`pipeline_gmdit.py` first), validate sample quality empirically, and roll back
if anything looks off.

---

## What to do if any new bug surfaces

The refactor is reversible:

1. **No regression possible on the existing path** as long as no caller passes
   explicit `alpha_*` kwargs (Test 6 proves bit-exact). Default behaviour is
   unchanged.
2. **VP path issues** can be isolated to the specific call site that opted in.
3. The legacy `gmflow_posterior_mean_jit` is never deleted.

---

## Files in this evidence bundle

- `01_posterior_rederivation.ipynb` — SymPy + numerical derivation (executed in-place; 21 cells, 0 errors)
- `posterior_rederivation.nb` — Mathematica completing-the-square proof
- `02_mathematica_crosscheck.md` — Human-readable Mathematica companion + verified-output appendix
- `_replay_nb.wls` — `wolframscript -file` driver that replays the `.nb` cells
- `_run_sympy.py` — Script form of the SymPy derivation (mirrors `.ipynb` Sections 2–5)
- `_run_numerical.py` — Script form of the numerical tests (mirrors `.ipynb` Sections 6–8)
- `_test_differential_equivalence.py` — Test 6 (bit-exact)
- `_test_vp_stability.py` — Test 7 (VP float32 stability — caveat #1)
- `_test_wrapper_dispatch.py` — Phase 2 wrapper dispatch tests
- `03_go_no_go.md` — this document

---

## Revision history

**2026-04-26 — cleanup pass (this revision).**

Triggered by the five-reviewer experiment in
`experiments/2026-04-24-five-reviewers/`. Three problems with the previous
version of this document were fixed:

1. **Test 3 row was wrong**. Previous text cited `|diff| = 1.79e-07` from
   "cell 18" of `01_posterior_rederivation.ipynb`. That cell was a broken
   duplicate that fired `AssertionError: Trig mismatch: 0.21555` when
   actually run (caught by reviewer `coder-1`). The `1.77e-07` figure does
   exist — it's the output of the same test in `_run_numerical.py`, just not
   in the cell the doc pointed at. The notebook has been rebuilt; cell 17 of
   the new notebook is the canonical source. Number is now exact, not
   summarised.

2. **`posterior_rederivation.nb` was unevaluated**. The notebook used a bare
   symbol `v` instead of `\[Nu]` in `logQ`, which made the load-bearing
   `FreeQ[missingTerm, μk]` check return **False** when run. Five reviewer
   reports all called the math correct without anyone running the notebook.
   Patched (`v → \[Nu]`); now produces the expected `0` / `True` outputs.
   Verified by `_replay_nb.wls`. Two CAS engines (Mathematica 14.3 + SymPy)
   now independently confirm the k-independence claim.

3. **Test 7 row implied production safety**. Five reviewers all flagged that
   `_test_vp_stability.py` exercises a non-production function with a
   `zeta_max` clamp absent from shipped code. Test 7 row now carries this
   caveat explicitly, and the clamp gap is listed as blocking caveat #1.

The Phase plan was also restructured. Original wording said "GO with phased
rollout" without entry criteria; rolled-out Phase 4 was treated as inevitable.
Now: Phase 1 stays, Phase 2 ships with four small additions, Phase 3 demoted
(can't live in JIT), Phase 4 blocked on three explicit gates. The math and
the bit-exact equivalence claim are unchanged — only the surrounding evidence
trail and rollout gates are tightened.

**2026-04-23 — original.**

Initial decision document. Verdict was "GO with phased rollout" based on
seven evidence layers; cleanup of 2026-04-26 found two (Test 3, Test 7) were
materially mis-stated.
