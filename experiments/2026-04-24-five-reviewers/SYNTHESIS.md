# Five-Reviewer Synthesis — GMFlow Posterior Refactor

**Date**: 2026-04-24
**Reviewers**: scholar-1 (math-physics), scholar-2 (Bayesian), coder-1 (code-explorer), coder-2 (critical), general (mixed)
**Setup**: each agent received the verbatim same prompt, worked alone, wrote one report. No agent saw any other agent's report.

---

## TL;DR

5/5 say **ship Phase 2** in some form. 5/5 say **defer Phase 4** until two specific gaps close. The reviewers split on whether the gaps are blocking or merely tightening — and one reviewer (coder-1) surfaced a **real integrity defect in the go/no-go evidence trail** that the others either missed or downplayed.

After ground-truth verification of contested claims, the highest-priority action is **fix the go/no-go doc and the notebook before any further phase work**.

---

## 1. Where all five agree

| Finding | Action |
|---|---|
| Symbolic derivation is correct (completing-the-square, Mathematica notebook). | Trust the math. |
| `vk·ν²/(2·dk)` truly cancels under softmax — but only because `gm_vars` is shape `(bs,*,1,1,1,1)` (shared across K). | Add a shape assertion in the JIT. |
| Bit-exactness on the linear path is genuine (operator-order mirroring on `aos*x/sigma`). | Phase 1 is safe; no rollback. |
| `_test_vp_stability.py` tests a *test-only* function with a `zeta_max` clamp; the shipped `gmflow_posterior_mean_jit_general` has no clamp. The go/no-go's "VP stable" evidence is therefore disconnected from shipped behaviour. | Either port the clamp into production, or rewrite the test to target the shipped function. |
| `piflow_policies/gmflow.py:73,81` imports `gmflow_posterior_mean_jit` directly and bypasses the wrapper. Phase 2 leaves it on the legacy code path. | Route policy through the wrapper, or document why it's locked. |
| Phase 4 must wait until (a) the clamp gap is closed, (b) at least one real-checkpoint run validates plumbing. | Block Phase 4 on these gates. |

This convergence — five independent agents, two different lenses each, all landing on the same five gaps — is strong evidence those gaps are real.

---

## 2. Where they disagree (and ground truth)

### Disagreement A — does cell 18 of `01_posterior_rederivation.ipynb` actually pass?

| Reviewer | Claim |
|---|---|
| **scholar-1** | Cited `\|diff\|=1.79e-7` from the go/no-go. Did **not** open cell 18. |
| **scholar-2** | Did not address Test 3 specifically. |
| **coder-1** | Cell 18 fails with `AssertionError: Trig mismatch: 0.21555089950561523` (analytic=1.697954, brute=1.913505). The go/no-go's `1.79e-7` figure does not appear in any cell. |
| **coder-2** | Did not catch this specific issue. |
| **general** | Acknowledged the `AssertionError` but framed it as a "noisy artifact" — per-channel marginal vs joint posterior weights coupled across channels. |

**Ground truth (verified by reading the notebook JSON directly):**

- Cell 17 (linear schedule): `(a) vs (b) max abs diff [should be ~0]: 1.19e-07 — PASS`.
- Cell 18 (trig schedule, **1D exact** test): `analytic=1.697954  brute=1.913505  |diff|=2.16e-01 — AssertionError`.
- The number `1.79e-07` is not present in any cell output.
- The cell-18 source comment warns that a *naive multi-channel* brute force would compute a different quantity. But the assertion that fired is from the **C=1 (single-channel) 1D exact** comparison, where the cell author intended the formulas to agree. So general's "per-channel vs joint coupling" defense applies only to the multi-channel case the assert prevented from running, not to the 1D case that actually failed.

**Verdict on the disagreement:** coder-1 is correct. The go/no-go's Test 3 row is materially wrong as stated. The trig/VP path has not been numerically verified at the C=1 level the doc claims. The plausible reading is that someone wrote the cell, hit the assert, never reconciled it, and wrote the go/no-go anyway — or the notebook drifted after the doc was written. Either way the evidence trail does not back up the claim.

scholar-1 fell into a known failure mode: trusting a summary doc instead of opening the underlying artifact. general saw the assert but rationalised it. Only coder-1 followed the citation chain end-to-end.

### Disagreement B — does the production function actually overflow under VP at small `t`?

| Reviewer | Claim |
|---|---|
| **coder-2** | Production lacks `zeta_max` clamp; will produce `inf`/`nan` at small `t`. *Theoretical claim.* |
| **scholar-1** | "could still overflow `nu`" — flagged as concern, not measured. |
| **scholar-2** | Test 7 demonstrates "VP would be safe **if** a clamp were added" — framed as conditional. |
| **general** | **Re-ran the production function** at `t ∈ {1e-3, 1e-5, 1e-10, 1e-20}` with `var=10` and small-var/large-mean configs. "In all cases output stayed finite" via inf/inf cancellation in the `out_means / denom` ratio. |
| **coder-1** | Theoretical: at `t=1e-5`, `zeta·var ≈ 4e9`, will overflow. Did not run. |

**Ground truth:** general's empirical claim was not independently re-verified in this synthesis pass — it requires running the production code. But it is the only claim with a *measurement* attached. Two implications:

1. The "production VP path silently overflows" framing (coder-2, scholar-1, coder-1) is plausible-sounding but contradicted by the one reviewer who actually ran the code.
2. Even if general is right that the output happens to stay finite, the mechanism is "inf/inf accidentally cancels in the right ratio" — that is fragile and unintentional. The clamp should still ship, not because it fixes a bug, but because relying on undocumented IEEE-754 cancellation is bad engineering hygiene.

This is a useful instance where four reviewers reasoned from theory and one reviewer ran the code. The theory-only reviewers were **directionally right but factually overstated** the severity.

### Disagreement C — is the wrapper's "fill missing alpha with 1−sigma" sensible?

scholar-1 says it is a foot-gun (silent schedule-mixing if a caller passes only one of the pair). general and scholar-2 don't flag it. coder-1 calls it an untested half-VP mode.

This is a genuine design ambiguity in `gmflow.py:213-216` that none of the existing tests exercise. scholar-1's recommendation (XOR-gate: require both alphas or neither) is the safer behaviour and costs almost nothing.

---

## 3. Novel concerns each reviewer uniquely surfaced

| Reviewer | Unique contribution |
|---|---|
| **scholar-1** | XOR-gate the wrapper dispatch; add Phase 3.5 (real-checkpoint linear bit-exactness vs previous git SHA). |
| **scholar-2** | Pin a torch version range or add bit-exact test to CI to guard against future JIT re-association. |
| **coder-1** | **The cell-18 / Test-3 evidence-trail integrity issue** (highest-impact catch). Phase 3's debug assertion cannot live inside `@torch.jit.script` — would need a Python-level wrapper. |
| **coder-2** | `gm_vars` cache mutation side-effect at `gmflow.py:209` (pre-existing but worth knowing). Negative-zeta path when `t > t_src` silently snaps to eps. Multi-channel `gm_vars` would silently break the formula. |
| **general** | **Empirically re-ran the test scripts and the production function** — only reviewer with measurement-grounded claims. Caught the standalone `~/Documents/piFlow` divergent operator order (relevant because user's MEMORY.md says submodule is canonical — confirms that). |

The two highest-value novel catches: coder-1's evidence-integrity finding, and general's act of actually running the code.

---

## 4. Reliability calibration

Sorted by what each reviewer caught vs missed:

1. **coder-1** — caught the only finding that nobody else caught (evidence-trail defect). Was theoretically wrong on the overflow severity (didn't run). **Highest unique signal.**
2. **general** — only one to run the production code. Empirically grounded. Downplayed cell-18 but acknowledged it. **Highest factual reliability.**
3. **scholar-2** — solid math, conservative recommendations, missed the cell-18 issue.
4. **coder-2** — strong on numeric-hazard taxonomy but two of three "critical" findings were theoretical and one (overflow) was empirically wrong.
5. **scholar-1** — strongest math write-up but trusted the go/no-go's `1.79e-7` figure without opening the cell. Classic "review the summary, not the source" miss.

The lesson for future experiments: **the willingness to actually run code or open the underlying artifact dominates pure analytical chops.** Two of the five reviewers (coder-1, general) did this; their reports surfaced the only findings that change what should happen next.

---

## 5. Composite verdict

**Phase 1 (shipped)**: stays. Math correct, bit-exact equivalence honest, no caller passes non-linear `alpha` yet so blast radius is zero.

**Phase 2 (wrapper)**: ship, but bundle these four small additions in the same PR:
- **a.** Tighten dispatch gate at `gmflow.py:211` to "both alphas or neither" (XOR foot-gun).
- **b.** Add a runtime shape assertion that `gm_vars`'s K and channel dims are both 1.
- **c.** Route `piflow_policies/gmflow.py:73,81` through the wrapper, even if it never passes alpha.
- **d.** Port the analytic `zeta_max` clamp from `_test_vp_stability.py:36` into `gmflow_posterior_mean_jit_general` so the test evidence matches shipped behaviour. (general's empirical safety result probably means this is hygiene rather than a bugfix, but ship the clamp anyway.)

**Phase 3 (debug allclose)**: low priority. coder-1 is right that it cannot live inside `@torch.jit.script`. If it ships, it has to be a Python-level wrapper, and honestly Test 6's bit-exact CI gate may already be sufficient.

**Phase 4 (real VP at a call site)**: blocked on:
- **e.** Fix the go/no-go doc: either reproduce a passing trig brute-force test or remove the misleading Test 3 row. The current `1.79e-07` is unsupported.
- **f.** Reconcile cell 18 of the notebook: investigate why analytic and brute-force differ by 0.22 in the C=1, K=4 trig case. Possibilities: brute-force grid resolution issue, formula bug, or the cell's importance-reweighting setup is genuinely a different quantity. Until this is resolved, the VP analytic formula has no numerical-validation evidence.
- **g.** One real-checkpoint linear-schedule run (cheap, decisive).

**Highest-priority next action: e + f** — the documentation defect must be addressed before any reviewer can trust the existing evidence trail going forward.

---

## 6. What this experiment proved about the methodology

- 5 independent reviewers converged on 6/6 of the major real findings.
- Disagreements were not between equally-valid views — in two cases (cell 18, overflow severity) one reviewer was clearly closer to ground truth than the others, and the disagreement was resolved by checking artifacts.
- The reviewers most likely to surface novel findings were the ones who **actually opened files and ran code**, not the ones who reasoned hardest from the documentation.
- Same prompt + diverse agent types produced genuinely different lenses. The mix (2 scholars + 2 coders + 1 general) was a good shape; pure replication would have lost the cell-18 catch.
