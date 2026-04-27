# scholar-2 review — schedule-agnostic GMFlow posterior refactor

## Summary

The information-form derivation is correct, the dropped `vk·ν²/(2·dk)` term
genuinely cancels under softmax **provided `vk` is shared across components**,
and the bit-exact equivalence is plausible because the new JIT function
literally reuses the same operator graph as the legacy one. The Go/No-Go's
five-layer evidence holds. My main reservations are scoped:
(a) the k-independence argument is conditional on a code-path invariant that
is asserted but never test-enforced; (b) the bit-exactness claim is contingent
on PyTorch's JIT not reordering an algebraically-equivalent rewrite; (c) one
of the five call sites (the piflow_policies path) is wired directly to the
legacy JIT and would not benefit from the wrapper-level dispatch in Phase 2.
None of these block Phase 2; they are guard-rails.

## Math check

Writing the un-normalised posterior in information form,
`log q(x_0) = -A x_0² + B x_0 + C` with `A = 1/(2vk) + ζ/2` and
`B = μk/vk + ν` (`logQ` block in `posterior_rederivation.nb`:135-148), the
exact log-normalizer is `−B²/(4A) + C` modulo a `−½ log(−2A)` term. The
Mathematica notebook computes `missingTerm = logZFull − ΔLogW` and asserts it
equals `vk·ν²/(2·dk)` (lines 217-244). That expression is manifestly free of
`μk`, so when `vk` (and hence `dk = vk·ζ + 1`) is shared across the K mixture
components, the missing term contributes the **same** additive constant to
every `log w_k` and is annihilated by softmax. The proof is sound.

The "free of μk" check via `FreeQ[missingTerm, μk]` (line 256-258) is a
necessary but in isolation insufficient guard — what actually matters is that
`vk` and `ζ` are k-invariant. `ζ` is k-invariant by construction (it depends
only on the schedule). `vk` k-invariance is a property of the **calling
code**, not of the derivation, and that is the load-bearing precondition
which `03_go_no_go.md` (line 19, 21) leans on.

I verified this invariant holds in production: `gmflow_ops.py` consistently
labels `gm_vars` as shape `(bs, *, 1, 1, 1, 1)` (e.g. lines 74, 164, 240,
320, 421) and computes it as `(gm_logstds * 2).exp()` from a logstds tensor
that carries a singleton K dim. The wrapper at
`gmflow.py:208` also enforces this. So the derivation's precondition is met
**today**. It is not, however, structurally enforced — a future model that
emits per-component `logstds` (shape `(bs,*,K,1,1,1)`) would silently break
the k-independence argument with no runtime check. (See Risks.)

The α/σ generalisation itself is purely a substitution: `ζ` and `ν` in the
notebook (lines 59-75) keep their information-form structure regardless of
the schedule. The boundary-limit check (`limT0` → `xt`,
`02_mathematica_crosscheck.md`:78-83) is the right sanity test and confirms
the formula reduces correctly at t→0 under VP.

## Engineering check

The "bit-exact" claim rests on the fact that
`gmflow_posterior_mean_jit_general` (`gmflow.py`:75-123) reproduces the
legacy function's ops in the **same order**: same `aos_t * x_t / sigma_t`
two-divide pattern (line 109 vs line 58), same `(gm_vars * zeta + 1)` denom
order (line 113 vs line 62), same `softmax` axis. Since under the linear
substitution `alpha_t_src = 1 − sigma_t_src` is computed by the **caller**
and passed in, while the legacy function recomputes it internally as
`1 − sigma_t_src` (line 51), the only floating-point hazard is the order in
which `1 − sigma_t_src` is evaluated. Both happen to be a single `1 − x` op
on the same input, so IEEE-754 determinism gives bit-identity. Test 6
(`_test_differential_equivalence.py`:115-139, 20 seeds × 5 shapes, fp32+fp64)
empirically confirms `max abs diff = 0.0`. The test loads code via SHA-256-
verified source extraction (lines 60-86), which is a nice touch — guarantees
the test runs against actual production source.

The `_test_vp_stability.py` test does **not** test the production function;
it tests a private copy that adds a `zeta_max` clamp (line 36). Production
`gmflow_posterior_mean_jit_general` lacks that clamp. So Test 7 demonstrates
"VP would be safe **if** a clamp were added", not "VP is safe today". That
is fine for Phase 1 (no caller passes non-linear α yet) but Phase 4 cannot
ship without either porting the clamp into production or proving the eps
clamps on `denom` are sufficient. The Go/No-Go does not flag this gap.

## Risks / blind spots

1. **k-independence is a runtime invariant, not a type-level guarantee.** No
   `assert gm_vars.size(channel_dim) == 1 and gm_vars.size(gm_dim) == 1`
   exists in the JIT body. Future models with per-component variance would
   silently produce wrong (but plausible-looking) posteriors. Recommend a
   one-line shape assertion at the wrapper boundary.
2. **Test 7 tests a non-production function.** The clamp it validates is not
   in `gmflow_posterior_mean_jit_general`. Phase 4 prerequisite.
3. **piflow_policies/gmflow.py:81** still calls `gmflow_posterior_mean_jit`
   directly. Phase 2's wrapper update only catches the
   `self.gmflow_posterior_mean` path. Either deprecate the direct call or
   route it through the wrapper, otherwise non-linear schedules cannot reach
   the policy code path.
4. **No checkpoint test.** The Go/No-Go acknowledges this (line 35) but
   doesn't make it a Phase-2 gate.
5. **Bit-exactness across torch versions / JIT optimisation passes.** Today's
   torch may fuse identically; a future torch could re-associate. Worth
   pinning a torch version range or adding the bit-exact test to CI.

## Rollout opinion

Phase 2 is appropriate next; I would adjust:

- **Phase 2a**: ship the wrapper, **plus** add a shape assertion that
  `gm_vars`'s K and channel dims are both 1. Cheap, makes the derivation
  precondition self-documenting.
- **Phase 2b**: route the piflow_policies call site through the wrapper too,
  even though it never passes α. Otherwise Phase 4 cannot reach it.
- **Phase 3** (debug-mode allclose) is good but I'd also add the bit-exact
  test to CI so a torch upgrade can't silently break the no-op claim.
- **Phase 4** must port the analytic `zeta_max` clamp (or formally prove the
  existing `denom.clamp(min=eps)` is sufficient for VP).

## Verdict

**Ship Phase 2 now**, with the two small additions above (shape assert,
policy-path routing). Phase 1 does not need revisiting. Defer Phase 4 until
the clamp gap is closed.
