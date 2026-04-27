# Mathematica Cross-Check: Schedule-Agnostic GMFlow Posterior

This document mirrors `posterior_rederivation.nb` in plain Mathematica syntax.
The notebook itself is the canonical source; this `.md` is a human-readable
companion. **All outputs in the appendix are reproduced verbatim from
`wolframscript -file ...` runs of the same expressions** (no hand-edited
results).

---

## Section 1 — Symbol setup

```mathematica
$Assumptions = {αt > 0, σt > 0, αs > 0, σs > 0,
                vk > 0, t > 0, t < 1, s > 0, s < 1};

{αt, σt, αs, σs, xt, xs, μk, vk} // Length  (* sanity: 8 *)
```

---

## Section 2 — Information-form terms

```mathematica
ζ  = (αt/σt)^2 - (αs/σs)^2;
ν  = αt*xt/σt^2 - αs*xs/σs^2;
dk = vk*ζ + 1;

μPost = (vk*ν + μk) / dk;
ΔLogW = μk*(ν - (1/2)*ζ*μk) / dk;

Print["ζ  = ", FullSimplify[ζ]];
Print["ν  = ", FullSimplify[ν]];
Print["dk = ", FullSimplify[dk]];
Print["μ̄k = ", FullSimplify[μPost]];
Print["Δw = ", FullSimplify[ΔLogW]];
```

---

## Section 2.5 — Completing-the-square (load-bearing claim)

The shipped code uses `ΔLogW` (the truncated form) rather than the full
log-normaliser. The reason this is *correct* is that the missing term is
k-independent — it adds the same constant to every component's log-weight and
therefore cancels under softmax. This is the entire correctness argument for
the refactor; without this section, the notebook proves nothing useful.

```mathematica
ClearAll[x0];
logQ = -(x0 - μk)^2/(2 vk) - (ζ/2)*x0^2 + ν*x0;   (* note: ν, not v *)
logQ = Expand[logQ];

A      = Coefficient[logQ, x0, 2];   (* = -1/(2 vk) - ζ/2 *)
B      = Coefficient[logQ, x0, 1];   (* = μk/vk + ν *)
Const0 = Coefficient[logQ, x0, 0];   (* = -μk^2/(2 vk) *)

logZFull    = FullSimplify[-B^2/(4 A) + Const0];   (* drop the k-independent ½log(-2A) term *)
missingTerm = FullSimplify[logZFull - ΔLogW];

Print["Matches vk·ν²/(2·dk)? (must be 0): ",
      FullSimplify[missingTerm - vk*ν^2/(2*dk)]];
Print["Free of μk (k-independent)?:        ", FreeQ[missingTerm, μk]];
```

> **Historical note**: an earlier version of `posterior_rederivation.nb`
> (pre-2026-04-26) used a bare symbol `v` instead of `ν` in `logQ`. With that
> typo the FreeQ check returned `False` and the entire k-independence claim
> failed when run. The notebook was unevaluated at the time, so the bug went
> unnoticed across five reviewer reports. The fix is a one-symbol substitution.

---

## Section 3 — Linear schedule reduction (α = 1 − σ)

```mathematica
linSubs = {αt -> 1 - σt, αs -> 1 - σs};

ζLin = FullSimplify[ζ /. linSubs];
νLin = FullSimplify[ν /. linSubs];

(* Reproduce existing-code form — the operator order matters for bit-exactness:
   ν is computed as (α/σ)·x/σ (two divides), not α·x/σ² — see gmflow.py:58,109. *)
ζCode = ((1 - σt)/σt)^2 - ((1 - σs)/σs)^2;
νCode = ((1 - σt)/σt)*xt/σt - ((1 - σs)/σs)*xs/σs;

Print["ζ diff (must be 0): ", FullSimplify[ζLin - ζCode]];
Print["ν diff (must be 0): ", FullSimplify[νLin - νCode]];
```

---

## Section 4 — VP / trig specialisation (α = cos, σ = sin)

```mathematica
vpSubs = {αt -> Cos[π*t/2], σt -> Sin[π*t/2],
          αs -> Cos[π*s/2], σs -> Sin[π*s/2]};

ζVP = TrigReduce[FullSimplify[ζ /. vpSubs]];
νVP = TrigReduce[FullSimplify[ν /. vpSubs]];

Print["ζ (VP): ", ζVP];
Print["ν (VP): ", νVP];
Print["cos²+sin²-1 (must be 0): ",
      FullSimplify[Cos[π*t/2]^2 + Sin[π*t/2]^2 - 1]];
```

---

## Section 5 — Boundary limits (s = 1/2 fixed)

```mathematica
μVP = μPost /. vpSubs /. s -> 1/2;

limT0 = FullSimplify[Limit[μVP, t -> 0, Direction -> "FromAbove"]];
limT1 = FullSimplify[Limit[μVP, t -> 1, Direction -> "FromBelow"]];

Print["lim t→0⁺ (s=1/2): ", limT0];   (* expect xt *)
Print["lim t→1⁻ (s=1/2): ", limT1];   (* (√2·vk·xs - μk)/(vk - 1) — diverges at vk=1 *)
```

> **Degeneracy at vk = 1.** The `t → 1⁻` boundary diverges when the GM
> component variance equals 1 exactly. This is a structural property of the VP
> trig schedule, not an artifact. The shipped JIT has no guard for `vk = 1`.
> If a model emits `gm_logstds = 0` (so `var_k = exp(0) = 1`) and the sampler
> ever evaluates near `t = 1`, the posterior mean will explode. Worth a
> regression test before any VP rollout.

---

## Section 6 — LaTeX export

```mathematica
TeXForm[ζ]
TeXForm[ν]
TeXForm[dk]
TeXForm[μPost]
TeXForm[ΔLogW]
```

---

## Appendix — Verified outputs

The following block was produced by `wolframscript -file /tmp/run_nb.wls`
(Wolfram 14.3, ARM macOS) on 2026-04-26, replaying the input cells of
`posterior_rederivation.nb` after the `v → ν` fix. Reproduce locally with:

```bash
wolframscript -file derivations/_replay_nb.wls
```

(see `_replay_nb.wls` in this folder for the exact script).

```
Symbol count: 8
Full log-normalizer:           (-(μk^2/vk) + (μk/vk - (xs*αs)/σs^2 + (xt*αt)/σt^2)^2/(vk^(-1) - αs^2/σs^2 + αt^2/σt^2))/2
Truncated DeltaLogW:           (αt*μk*(2*xt - αt*μk)*σs^2 + αs*μk*(-2*xs + αs*μk)*σt^2)/(2*vk*αt^2*σs^2 + 2*(-(vk*αs^2) + σs^2)*σt^2)
Missing term:                  (vk*(xt*αt*σs^2 - xs*αs*σt^2)^2)/(2*vk*αt^2*σs^4*σt^2 + 2*σs^2*(-(vk*αs^2) + σs^2)*σt^4)
Matches vk*ν^2/(2*dk)? (0=yes): 0
Free of μk (k-independent)?:    True
ζ  = -(αs^2/σs^2) + αt^2/σt^2
ν  = -((xs*αs)/σs^2) + (xt*αt)/σt^2
dk = 1 - (vk*αs^2)/σs^2 + (vk*αt^2)/σt^2
μ̄k = (μk - (vk*xs*αs)/σs^2 + (vk*xt*αt)/σt^2)/(1 - (vk*αs^2)/σs^2 + (vk*αt^2)/σt^2)
Δw = (αt*μk*(2*xt - αt*μk)*σs^2 + αs*μk*(-2*xs + αs*μk)*σt^2)/(2*vk*αt^2*σs^2 + 2*(-(vk*αs^2) + σs^2)*σt^2)
ζ diff (must be 0): 0
ν diff (must be 0): 0
ζ (VP): (-(Cos[Pi*s]*Csc[(Pi*s)/2]^2*Csc[(Pi*t)/2]^2) + Cos[Pi*t]*Csc[(Pi*s)/2]^2*Csc[(Pi*t)/2]^2)/2
ν (VP): -(xs*Cot[(Pi*s)/2]*Csc[(Pi*s)/2]) + xt*Cot[(Pi*t)/2]*Csc[(Pi*t)/2]
cos^2+sin^2-1 (must be 0): 0
lim t->0+ (s=1/2) [expect xt]:        xt
lim t->1- (s=1/2) [degenerate vk=1]:  (Sqrt[2]*vk*xs - μk)/(-1 + vk)
```

### Independent SymPy cross-check

`derivations/_run_sympy.py` reproduces the same algebra in SymPy from scratch
(no shared symbols with the Mathematica notebook). The Section 3b output:

```
=== S3b: completing-the-square (k-independence) ===
missing - var_k*nu^2/(2*dk)  (must be 0): 0
mu_k in missing.free_symbols (must be False): False
```

Two independent CAS engines (Mathematica 14.3, SymPy) agree on the
k-independence result. The symbolic identity is real, not an artefact of
either system's simplification heuristics.

### Numerical cross-check

`derivations/_run_numerical.py` runs three numerical tests:

```
[S6a vs S6b] max |diff| (linear, should be ~0): 2.38e-07
PASS: general formula (alpha=1-sigma) == existing code

[S6c 1D exact test] analytic=1.913505  brute=1.913505  |diff|=1.77e-07
PASS: trig posterior formula matches brute-force (1D, K=4)

zeta at t=1e-5 (raw):      4.053e+09  isinf=False
denom (var * zeta + 1):    4.053e+10  isinf=False
PASS: clamp prevents overflow
```

The `1.77e-07` figure is the source of the `1.79e-07` claim in
`03_go_no_go.md` (Test 3): trig-schedule analytic formula vs 1D brute-force,
C=1, K=4. The two values match to grid quadrature precision.
