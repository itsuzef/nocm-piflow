# GMFlow Schedule-Agnostic Posterior — Rollout Plan

**Subject**: Generalising `gmflow_posterior_mean_jit` from the hardcoded linear
schedule (`α = 1 − σ`) to an arbitrary `(α, σ)` pair, then opting call sites
in one at a time.

**Replaces**: `03_go_no_go.md` (verdict-shaped doc, retired 2026-04-26).
**Companion docs**: `01_posterior_rederivation.ipynb`, `02_mathematica_crosscheck.md`,
`posterior_rederivation.nb`, `experiments/2026-04-24-five-reviewers/SYNTHESIS.md`.

---

## How to read this document

This is an **executable plan**, not a decision memo. Each phase has explicit
entry criteria, work items with file/line anchors, verification commands, exit
criteria, and a rollback. **Do not start any phase whose entry criteria are
unchecked.**

Items tagged **[VERIFY FIRST]** are claims this plan currently *relies on* but
that have **not been re-confirmed in this working tree at this revision**. They
must pass before the Phase 2 PR is opened. Items tagged **[BLOCKING]** must
close before the phase they gate; **[HYGIENE]** items should ship in the same
PR but are not strict pre-conditions.

---

## 0. Status snapshot (what is true *right now*)

| Item | State |
|---|---|
| `gmflow_posterior_mean_jit_general` exists alongside the legacy JIT (`repos/piFlow/lakonlab/models/diffusions/gmflow.py:76–123`) | Shipped (Phase 1). |
| Wrapper `GMFlowMixin.gmflow_posterior_mean` accepts optional `alpha_t_src` / `alpha_t` and dispatches via `use_general` flag (same file, lines 169–230) | Shipped (Phase 2 plumbing in place; hygiene additions still pending). |
| `gm_vars` shape assertion inside the general JIT | **Not done.** |
| XOR-gate on the wrapper's `(alpha_t_src, alpha_t)` pair | **Not done.** Wrapper currently silently fills the missing one with `1 − sigma_*` (lines 213–216). |
| `piflow_policies/gmflow.py:73,81` routed through the wrapper | **Not done.** Still calls `gmflow_posterior_mean_jit` directly. |
| `zeta_max` clamp ported from `_test_vp_stability.py:36` into production | **Not done.** Production has no clamp. |
| Cell 17 of `01_posterior_rederivation.ipynb` shows the C=1, K=4 trig brute-force vs analytic check | Claimed `|diff| ≈ 1.77e-07` after the 2026-04-26 notebook rebuild. **[VERIFY FIRST]** — see V1. |
| Mathematica notebook produces `missingTerm = 0` and `FreeQ[missingTerm, μk] == True` | Claimed after the `v → \[Nu]` patch. **[VERIFY FIRST]** — see V2. |
| Bit-exact equivalence (`max abs diff = 0.0`, 20 seeds × 5 configs × fp32+fp64) | Last reported PASS. **[VERIFY FIRST]** — see V3. |
| Production VP path stays finite at small `t` without the clamp | One reviewer measured this; not independently re-run. **[VERIFY FIRST]** — see V5. |

If any cell of this table changes, re-validate every downstream phase before
acting on it.

---

## 1. Pre-execution verification gates  **[VERIFY FIRST]**

Run all eight gates from a clean shell at the current working-tree SHA. Capture
stdout/stderr. **All eight must pass** before opening the Phase 2 PR.

| ID | Gate | Why it matters | Command | Pass condition |
|---|---|---|---|---|
| **V1** | Re-run `01_posterior_rederivation.ipynb` end-to-end. | The notebook was patched after `coder-1` caught a broken cell (former cell 18). The doc trail relies on the rebuilt cell 17 actually executing cleanly. | `jupyter nbconvert --to notebook --execute --inplace derivations/01_posterior_rederivation.ipynb` | Exit 0; cell 17 prints `analytic = 1.913505`, `brute = 1.913505`, `|diff| ≤ 5e-7`. |
| **V2** | Re-run the Mathematica replay. | The `.nb` had a bare-symbol bug (`v` vs `\[Nu]`) that made `FreeQ[missingTerm, μk]` return `False` when actually run. Five reviewers missed this. Re-run in a fresh kernel. | `wolframscript -file derivations/_replay_nb.wls` | `missingTerm` simplifies to `0`; `FreeQ[missingTerm, μk]` returns `True`. |
| **V3** | Bit-exact equivalence sweep. | Phase 2's "zero blast radius" guarantee depends on this number being literally `0.0`, not just within float epsilon. | `python derivations/_test_differential_equivalence.py` | `max abs diff = 0.0` for every (seed × shape × dtype) cell. |
| **V4** | Wrapper-dispatch source audit + functional equivalence. | Confirms the four opt-in routes (default, both alphas, only `alpha_t_src`, only `alpha_t`) all reduce to bit-identical output under linear schedule, and the source still contains the dispatch tokens. | `python derivations/_test_wrapper_dispatch.py` | Source audit clean; all four routes equivalent (`< 1e-5` fp32, `< 1e-12` fp64); VP smoke finite. |
| **V5** | Empirical VP-stability re-check on the **production** function (no `zeta_max` clamp). | Only one reviewer (`general`) measured this; SYNTHESIS marks the result "not independently re-verified". If this fails, caveat C1 escalates from blocking-Phase-4 to blocking-Phase-2. | New script (write as `derivations/_test_production_vp_finite.py`): import `gmflow_posterior_mean_jit_general` via the same regex/importlib trick as `_test_wrapper_dispatch.py`; sweep `t ∈ {1e-3, 1e-5, 1e-10, 1e-20}` × `var_k ∈ {1.0, 5.0, 10.0}` × `s ∈ {0.5, 0.9}`; assert `torch.isfinite(out).all()`. | All combinations finite. Record `max(out.abs())` and `(out_means / denom)` ratios for the audit log. |
| **V6** | Submodule SHA pin. | Tests run against `repos/piFlow/lakonlab/models/diffusions/gmflow.py`. If the submodule moved since the evidence trail was last validated, every gate above is talking about a different file. | `git -C repos/piFlow rev-parse HEAD` and compare to the SHA recorded in the last evidence run. | SHAs match; if not, re-run V1–V5 against the new SHA before proceeding. |
| **V7** | Call-site audit. | Phase 2's "5 call sites, all linear" claim is the basis for the bit-exactness blast-radius argument. | `rg -n 'gmflow_posterior_mean' repos/piFlow -t py` and dedupe defs from calls. | Exactly the call sites listed in §2.3 below; if more, update the audit and re-evaluate hygiene item C3. |
| **V8** | Script-form parity. | `_run_sympy.py` and `_run_numerical.py` are scripted mirrors of the notebook. They are independent corroboration for V1 — if the notebook fails but the scripts pass (or vice versa), the truth lives in whichever path produces consistent numbers, not in the doc. | `python derivations/_run_sympy.py && python derivations/_run_numerical.py` | Both exit 0. The trig-schedule numerical block (`Section 6c` of `_run_numerical.py`) prints `|diff| ≤ 5e-7`. |

**If any gate fails**: fix the underlying issue, update §0, re-run *all* gates
(no partial re-runs — failures often invalidate adjacent assumptions), then
proceed.

---

## 2. Phase 2 — Wrapper dispatch + hygiene bundle  **(ready to execute once V1–V8 pass)**

### 2.1 Entry criteria
- [ ] V1–V8 all green at the current submodule SHA.
- [ ] §0 status table updated to reflect the green run.
- [ ] Reviewer assigned (this PR is the load-bearing one — math correctness has been argued; what's being reviewed is the *engineering hygiene*).

### 2.2 Work items (single PR, in this order)

**C1 — Port the analytic `zeta_max` clamp into production.** **[BLOCKING gate (e) for Phase 4; HYGIENE for Phase 2]**

- File: `repos/piFlow/lakonlab/models/diffusions/gmflow.py:108`.
- Source pattern: `_test_vp_stability.py:36` (`zeta = zeta.clamp(min=-zeta_max, max=zeta_max)`).
- Add a `zeta_max: float` argument to `gmflow_posterior_mean_jit_general` with default derived from `torch.finfo(zeta.dtype).max / max_var_assumed` (see `_test_vp_stability.py:73–74`). Plumb the default through the wrapper.
- **[VERIFY FIRST]** Decide whether `max_var_assumed` is a constant (`10.0` per the test) or read from `gm_vars.max()` at runtime. The test uses a constant; running off `gm_vars.max()` is more defensive but breaks JIT determinism if `gm_vars` is data-dependent. Default position: constant `10.0`, documented assumption, asserted via C2.
- After porting, fold `_test_vp_stability.py` to target the *production* function (drop the locally-defined copy at lines 19–47). The test must continue to pass.

**C2 — Assert `gm_vars` shape inside the general JIT.** **[HYGIENE]**

- File: `repos/piFlow/lakonlab/models/diffusions/gmflow.py:111` (after `nu = nu.unsqueeze(gm_dim)`).
- Add `torch._assert(gm_vars.size(gm_dim) == 1, "gm_vars must be shared across mixture components")` and a matching assertion that `gm_vars.size(channel_dim) == 1`.
- Mirror in `gmflow_posterior_mean_jit` for symmetry, even though the legacy path is provably-shared today.
- Add a unit test (`derivations/_test_gm_vars_shape_assert.py`) that constructs per-component `gm_vars` and asserts the JIT raises.

**C3 — XOR-gate the wrapper dispatch.** **[HYGIENE]**

- File: `repos/piFlow/lakonlab/models/diffusions/gmflow.py:211–216`.
- Replace the silent `1 − sigma_*` auto-fill with: if exactly one of `(alpha_t_src, alpha_t)` is `None`, raise `ValueError("Pass both alpha_t_src and alpha_t, or neither.")`.
- Update `_test_wrapper_dispatch.py:_reference_dispatch` to mirror (only `(both, neither)` remain valid; the `only_t_src` and `only_t` cases now expect a raise — convert them into `pytest.raises`-style assertions).
- **[VERIFY FIRST]** Confirm no caller in the current tree relies on the half-fill behaviour. Run `rg -n 'gmflow_posterior_mean\(' repos/piFlow -t py` and inspect every site for `alpha_t=`/`alpha_t_src=` kwargs. **As of writing, no call site passes either alpha kwarg, so the XOR gate is safe.**

**C4 — Route `piflow_policies/gmflow.py:73,81` through the wrapper.** **[HYGIENE]**

- File: `repos/piFlow/lakonlab/models/diffusions/piflow_policies/gmflow.py:71–86`.
- Replace direct `gmflow_posterior_mean_jit(...)` calls with `self.gmflow_posterior_mean(...)` (or whatever wrapper entrypoint the policy class can reach). The policy holds `sigma_t_src`, `x_t_src`, `eps`, the GM dict — all the arguments the wrapper needs.
- **[VERIFY FIRST]** Confirm the policy class has access to a `gmflow_posterior_mean`-shaped wrapper. If it does not (policies don't subclass `GMFlowMixin`), either (a) inject the bound method, or (b) document why this site is intentionally schedule-locked and add a comment with a link to this doc. Default position: (b), because the policy is performance-critical and the wrapper adds dispatch overhead.
- If (a) is chosen, add a policy-level test: same setup as the existing wrapper-dispatch test, asserting bit-exactness against the legacy direct call.

### 2.3 Call-site inventory (audit basis for C3 and C4)

Five sites identified at the SHA in §0; **re-confirm via V7 before merging**:

| File | Line(s) | Kind |
|---|---|---|
| `repos/piFlow/lakonlab/models/diffusions/gmflow.py` | `217` (general path), `221` (legacy path) | Wrapper dispatch (the *target* of this refactor — not a "call site" in the audit sense). |
| `repos/piFlow/lakonlab/models/diffusions/gmflow.py` | `340–341` | `gm_2nd_order` calls `self.gmflow_posterior_mean(...)`. Routes through wrapper. |
| `repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py` | (see V7 output) | Pipeline-level call. Routes through wrapper. |
| `repos/piFlow/lakonlab/models/diffusions/piflow_policies/gmflow.py` | `73`, `81` | **Bypasses wrapper.** Target of C4. |

If V7 surfaces any new site, treat the merge as gated until it is classified
the same way.

### 2.4 Verification (run from a clean shell, in order)

```bash
python derivations/_test_differential_equivalence.py     # V3 still PASS
python derivations/_test_wrapper_dispatch.py             # V4, with C3 asserts now expecting raises
python derivations/_test_vp_stability.py                  # now exercises *production* function (post-C1)
python derivations/_test_gm_vars_shape_assert.py          # new in C2
python derivations/_test_production_vp_finite.py          # new in V5; must still PASS post-C1
```

### 2.5 Acceptance / exit criteria

- [ ] All five test scripts exit 0.
- [ ] `rg gmflow_posterior_mean repos/piFlow -t py` shows the policy site routed through the wrapper (or comment-documented per C4 fallback).
- [ ] No caller passes only one of the alpha pair (XOR gate would raise).
- [ ] `gm_vars` per-component shape raises a JIT-level assert.
- [ ] CI run on the submodule's pipeline regression shows no behavioural change on the linear path.

### 2.6 Rollback

Phase 2 is a single PR. To roll back: revert the PR. Because no caller passes
`alpha_t_src` or `alpha_t` yet (V7), reverting C3 + C4 cannot regress any
existing call. Reverting C1 only matters if a caller *also* passed alpha
kwargs — won't exist until Phase 4. C2 is a pure assertion; reverting it
removes a safety net but cannot break anything that wasn't already broken.

---

## 3. Phase 3 — Debug allclose  **(redesigned; low priority)**

The originally-proposed JIT-internal `if alpha ≈ 1 - sigma: assert allclose(legacy)`
**is not implementable**: `@torch.jit.script` cannot read environment variables,
cannot do Python-level conditional imports, and cannot call back into a sibling
JIT function with a different signature. This was caught by `coder-1`.

### 3.1 Decision: drop unless someone hits a divergence in practice

V3 (bit-exact CI gate) already provides the equivalence check Phase 3 was meant
to provide. Shipping Phase 3 adds maintenance burden for no measurement-grounded
benefit.

### 3.2 If we ever want it back (redesign)

Wrap, don't embed. Add a Python-level helper in `gmflow.py`:

```python
def _gmflow_posterior_mean_check(*args, **kwargs):
    if not os.environ.get('GMFLOW_DEBUG'):
        return gmflow_posterior_mean_jit_general(*args, **kwargs)
    out_general = gmflow_posterior_mean_jit_general(*args, **kwargs)
    if _is_linear_schedule(args):
        out_legacy = gmflow_posterior_mean_jit(*_legacy_args(args, kwargs))
        torch.testing.assert_close(out_general, out_legacy, atol=0, rtol=0)
    return out_general
```

This lives entirely outside any `@torch.jit.script` boundary. Cost: a Python
dispatch on every call when `GMFLOW_DEBUG=1`. Acceptable for debug, never on
in prod.

---

## 4. Phase 4 — Schedule-aware sampler at a live call site  **(BLOCKED)**

### 4.1 Entry criteria (gates e/f/g — all three must close)

- [ ] **Gate (e)** — C1 merged: production `gmflow_posterior_mean_jit_general` has the `zeta_max` clamp. (Same item as Phase 2 work item C1.)
- [ ] **Gate (f)** — `var_k = 1` regression test exists and passes. **Not yet written.** Spec:
  - File: `derivations/_test_vp_var1_boundary.py`.
  - Setup: VP schedule, `t ∈ {0.999, 1 − 1e-3, 1 − 1e-5}`, `var_k ∈ {0.99, 1.00, 1.01}`.
  - Assertion: output finite for `var_k ≠ 1`; documented behaviour (clamp / NaN / explicit error) at `var_k = 1` exactly.
  - **[VERIFY FIRST]** the analytic divergence at `var_k = 1` (`(√2·var_k·x_s − μ_k)/(var_k − 1)`) is reproducible by SymPy + Mathematica before writing the test. Both confirmed it once; re-confirm at the rebuild SHA.
- [ ] **Gate (g)** — Real-checkpoint linear-schedule equivalence run. **Not yet executed.** Spec:
  - Pick a checkpoint that exercises the GMFlow posterior path (the research path in `pipeline_gmdit.py` is the natural candidate).
  - Run a fixed-seed sampling job at the previous git SHA → record outputs to disk.
  - Run the same job at the post-Phase-2 SHA → assert `max abs diff = 0.0` against the saved outputs.
  - Synthetic-tensor blind spot (caveat C6 from the old doc): real outputs may have extreme means or near-degenerate weights that synthetic `randn` doesn't exercise. This is the only way to find out.

### 4.2 Per-call-site rollout (once entry criteria pass)

Opt in **one site at a time**. Suggested order:

1. **`pipeline_gmdit.py` research path.** Lowest blast radius (research, not
   shipped to users). Sample-quality eval on a fixed prompt set; A/B against
   the linear schedule.
2. **`gm_2nd_order` second-order corrector** (`gmflow.py:340`). Higher impact.
   Only opt in if step 1 shows no quality regression.
3. **`piflow_policies/gmflow.py`** (only if C4(a) was chosen — i.e., the
   policy now routes through the wrapper). Highest impact (sampling-time
   policy). Last.

For each site:
- Add explicit `alpha_t_src=` / `alpha_t=` to the call (XOR gate enforces both).
- Run gate (g) again, scoped to this site.
- Sample-quality eval on a frozen prompt/seed set, compared against the
  linear-schedule baseline.
- Decision: ship / hold / roll back, recorded in this doc.

### 4.3 Rollback

Per-site revert. Because each opt-in is a single-line change at a call site,
reverting is mechanical. The legacy `gmflow_posterior_mean_jit` is **never
deleted** — it remains the bit-exact fallback for the rest of the codebase.

---

## 5. Reversibility / kill switch

The whole refactor is reversible in three layers:

1. **Default-path safety.** No caller passes `alpha_*` kwargs until Phase 4
   (V7 enforces this). Default behaviour goes through the legacy JIT,
   bit-exactly.
2. **Per-site rollback** (Phase 4): one-line revert per call site.
3. **Whole-feature kill switch** (last resort): revert the Phase 2 PR. The
   legacy JIT is untouched throughout.

---

## 6. Open questions / decisions outstanding

These do not block Phase 2 but should be resolved before Phase 4 starts.

| # | Question | Default position |
|---|---|---|
| Q1 | Is `max_var_assumed` for `zeta_max` a constant (`10.0`) or runtime-derived from `gm_vars.max()`? | Constant. C2's shape assert + a comment documenting the assumption. Runtime-derived breaks JIT determinism. |
| Q2 | When the policy class can't reach `GMFlowMixin.gmflow_posterior_mean` (the (a)-vs-(b) choice in C4), do we inject the bound method or leave the policy schedule-locked? | (b) — leave it locked, document, link to this plan. The policy is performance-sensitive and the dispatch overhead is non-zero. |
| Q3 | What is the test budget for the real-checkpoint run (gate g)? Number of seeds, prompts, sampling steps. | Single fixed seed × 8 prompts × default sampling steps. Bit-exactness is binary — one diverging element kills the whole gate, so a small budget is sufficient. |
| Q4 | Mathematica + SymPy disagreed at one point; both now converge after the `v → \[Nu]` patch. Should we add a CI gate that re-runs `_replay_nb.wls` on every change to the `.nb`? | Yes — costs nothing, prevents the failure mode that caused the original doc-integrity defect. Add as a follow-up after Phase 2. |

---

## 7. Files in this evidence bundle

| Path | Role |
|---|---|
| `01_posterior_rederivation.ipynb` | SymPy + numerical derivation. Cell 17 is the load-bearing trig-schedule check (V1). |
| `posterior_rederivation.nb` | Mathematica completing-the-square proof. Re-runnable via V2. |
| `02_mathematica_crosscheck.md` | Human-readable Mathematica companion + verified-output appendix. |
| `_replay_nb.wls` | `wolframscript -file` driver for V2. |
| `_run_sympy.py` | Script form of the SymPy derivation. V8 driver. |
| `_run_numerical.py` | Script form of the numerical tests. V8 driver. |
| `_test_differential_equivalence.py` | V3 driver (bit-exactness). |
| `_test_vp_stability.py` | Test 7 (currently exercises a test-only function; folds into V5 / production after C1). |
| `_test_wrapper_dispatch.py` | V4 driver (wrapper dispatch + source audit). |
| `_test_production_vp_finite.py` | V5 driver. **Written 2026-04-26.** Currently passes against the unclamped production function (general's empirical claim re-verified at SHA `388cdcb5…`). Forward-compatible with C1 via `zeta_max=inf` plumbing. |
| `_test_gm_vars_shape_assert.py` | C2 driver. **Written 2026-04-26.** Currently *expected to FAIL* — happy path passes, malformed-shape paths produce no error because the runtime assertion is not yet in production. C2 closes when this script exits 0. |
| `_test_vp_var1_boundary.py` | Phase 4 gate (f) driver. **Written 2026-04-26.** Hard assertions pass at the current SHA. Records the boundary baseline (`var_k=1`, `t→1`): output stays finite but grows to `\|out\| max ≈ 7.4e6` at `t=0.99999`, capped by `denom.clamp(min=eps)`. Follow-up: the `var_k > 1` regime also produces large outputs (negative `denom` snaps to `+eps`); not in gate (f) scope but worth a separate hazard note. |
| `03_rollout_plan.md` | This document. |
| `experiments/2026-04-24-five-reviewers/SYNTHESIS.md` | Five-reviewer synthesis; the source of every `[VERIFY FIRST]` flag below the math layer. |

---

## 8. Lineage

This document replaces `03_go_no_go.md` (renamed via `git mv` 2026-04-26).
The verdict-shaped predecessor is preserved in git history on the same path
prior to that commit. Substance differences:

- The verdict (GO Phase 1 / Conditional GO Phase 2 / HOLD Phase 4) is
  preserved as state in §0 and as entry criteria in §2.1, §4.1 — but
  reframed as **execution conditions**, not a decision.
- Every claim that was previously asserted as "we verified this" but rested
  on a single un-replicated run is now flagged **[VERIFY FIRST]** with a
  named gate (V1–V8).
- The Phase 3 demotion is recorded as a design change (§3), not buried in a
  caveat.
- Caveats that were "blocking Phase 4" in the old doc are now
  Phase 4 entry criteria (gates e/f/g) with concrete test specs.
- The original revision history is in git; not duplicated here.
