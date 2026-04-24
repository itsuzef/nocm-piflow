# Mathematica Cross-Check: Schedule-Agnostic GMFlow Posterior

Paste each block into a new cell (`Shift+Enter` to evaluate).

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

## Section 3 — Linear schedule reduction (α = 1 − σ)

```mathematica
linSubs = {αt -> 1 - σt, αs -> 1 - σs};

ζLin = FullSimplify[ζ /. linSubs];
νLin = FullSimplify[ν /. linSubs];

(* Reproduce existing-code form *)
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

(* Pythagorean identity — must be 0 *)
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
Print["lim t→1⁻ (s=1/2): ", limT1];   (* degenerate: t>s violates sampling order *)
```

---

## Section 6 — LaTeX export

```mathematica
TeXForm[ζ]
TeXForm[ν]
TeXForm[dk]
TeXForm[μPost]
TeXForm[ΔLogW]
```
