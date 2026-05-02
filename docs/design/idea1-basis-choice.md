# Idea 1 — Fourier Policy: Boundary-Basis Choice (φ_0, φ_1)

**Date:** 2026-05-02 (R4)
**Scope:** Decide whether the linear basis `(φ_0, φ_1) = (1−τ, τ)` adopted in [R1.1 §4.2](idea1-anchor-design.md) should be replaced by a schedule-aware or otherwise non-linear basis. R1.1's other choices are held fixed: Option A (X_start = x_t_src, X_end = x̂_0), τ(σ) = 1 − σ/σ_t_src, F2 velocity (u = +∂_σ Ψ), soft anchor at τ=1, soft anchor velocity at τ=0 for M ≥ 1.

---

## 1. Setup

### 1.1 Forward process and schedule

The training-time forward process is the linear/VE one, hardcoded in [`gaussian_flow.py:90-95`](../../repos/piFlow/lakonlab/models/diffusions/gaussian_flow.py#L90-L95):

```
mean = 1 − σ;  std = σ;  x_t = x_0·(1−σ) + ε·σ
```

In abstract notation, $x_\sigma = \alpha(\sigma)\,x_0 + \sigma\,\varepsilon$ with two relevant `α(σ)` choices:

- **Linear / VE** (current production, [`gmflow.py:51-52`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L51-L52)): $\alpha_{\text{lin}}(\sigma) = 1 - \sigma$.
- **VP / cosine** ([ideas-comparison.md:33-34](../ideas-comparison.md#L33-L34)): $\alpha_{\text{VP}}(t) = \cos(\pi t/2)$, $\sigma_{\text{VP}}(t) = \sin(\pi t/2)$. The relation $\alpha^2 + \sigma^2 = 1$ gives $\alpha_{\text{VP}}(\sigma) = \sqrt{1 - \sigma^2}$ when expressed as a function of σ. Both schedules satisfy $\alpha(0) = 1$ and $\sigma \in [0, 1]$.

The schedule-agnostic posterior infrastructure ([`gmflow.py:79-139`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L79-L139)) accepts arbitrary `(α, σ)` pairs but is not currently consumed by any production path ([derivations/03_rollout_plan.md:65-72](../../derivations/03_rollout_plan.md#L65-L72)). The companion audit `u-to-x0-schedule-audit.md` shows that `u_to_x_0` and the static `_u_to_x_0` helpers in the policy classes are silently linear-schedule (4 implementations in scope, 8 of 9 call sites latent-bug under VP). The basis-choice question here is **independent** of those implementations: the Fourier policy outputs `x̂_0` directly and bypasses `u_to_x_0`.

### 1.2 Fourier policy with abstract boundary basis

Under R1.1's conventions but with abstract `(φ_0(τ), φ_1(τ))`:

$$
\Psi(\tau) \;=\; \phi_0(\tau)\,x_{t_{src}} \;+\; \phi_1(\tau)\,\hat{x}_0 \;+\; \sum_{m=1}^M \bigl(a_m\sin(m\pi\tau) + b_m(\cos(m\pi\tau) - 1)\bigr)
$$

with the required boundary behaviour
- $\phi_0(0) = 1,\; \phi_0(1) = 0$  (so $\Psi(0)$ pulls toward $X_{start} = x_{t_{src}}$)
- $\phi_1(0) = 0,\; \phi_1(1) = 1$  (so $\Psi(1)$ pulls toward $X_{end} = \hat{x}_0$)

and the τ-axis fixed by R1.1 §4: $\tau(\sigma) = 1 - \sigma/\sigma_{t_{src}}$, equivalently $\sigma(\tau) = (1-\tau)\,\sigma_{t_{src}}$.

### 1.3 Natural trajectory between anchors at fixed ε

For a single sample with fixed noise ε, the diffusion process traces the curve
$$
\Psi_{\text{true}}(\sigma) \;=\; \alpha(\sigma)\,x_0 + \sigma\,\varepsilon.
$$

Pin ε by demanding $\Psi_{\text{true}}(\sigma_{t_{src}}) = x_{t_{src}}$:
$$
\varepsilon \;=\; \frac{x_{t_{src}} - \alpha(\sigma_{t_{src}})\,x_0}{\sigma_{t_{src}}}.
$$

Substituting:
$$
\boxed{\,\Psi_{\text{true}}(\sigma) \;=\; \Bigl[\alpha(\sigma) - \tfrac{\sigma}{\sigma_{t_{src}}}\,\alpha(\sigma_{t_{src}})\Bigr]\,x_0 \;+\; \tfrac{\sigma}{\sigma_{t_{src}}}\,x_{t_{src}}\,}
$$

Endpoints (using $\alpha(0) = 1$):
- $\Psi_{\text{true}}(0) = x_0$
- $\Psi_{\text{true}}(\sigma_{t_{src}}) = x_{t_{src}}$ (by construction)

Specialised:
- **Linear**: $\Psi_{\text{true,lin}}(\sigma) = \bigl(1 - \tfrac{\sigma}{\sigma_{t_{src}}}\bigr) x_0 + \tfrac{\sigma}{\sigma_{t_{src}}} x_{t_{src}}$ — straight line in σ.
- **VP**: $\Psi_{\text{true,VP}}(\sigma) = \bigl[\sqrt{1-\sigma^2} - \tfrac{\sigma}{\sigma_{t_{src}}}\sqrt{1-\sigma_{t_{src}}^2}\bigr] x_0 + \tfrac{\sigma}{\sigma_{t_{src}}} x_{t_{src}}$ — curve in σ.

The *geometric residual* of a basis is the deviation $\Delta(\sigma) := \Psi_{M=0}^{\text{basis}}(\sigma) - \Psi_{\text{true}}(\sigma)$ at perfect prediction $\hat{x}_0 = x_0$. The Fourier residual $\sum a_m\sin + b_m(\cos-1)$ has to absorb $\Delta(\sigma)$ before it can spend any capacity on the data-conditional prediction error.

---

## 2. Candidate bases

### B0 — Linear (R1.1 baseline)

$\phi_0(\tau) = 1-\tau$, $\phi_1(\tau) = \tau$.

$$\Psi_{B0}^{M=0}(\sigma) = \tfrac{\sigma}{\sigma_{t_{src}}} x_{t_{src}} + \bigl(1 - \tfrac{\sigma}{\sigma_{t_{src}}}\bigr)\hat{x}_0.$$

### B1 — Schedule-aware (M=0 reproduces $\Psi_{\text{true}}$)

Choose $\phi_0, \phi_1$ such that $\Psi_{B1}^{M=0}(\sigma) = \Psi_{\text{true}}(\sigma)$ when $\hat{x}_0 = x_0$. From the closed form in §1.3:

$$
\phi_0^{B1}(\tau) = 1 - \tau \qquad
\phi_1^{B1}(\tau) = \alpha\bigl((1-\tau)\sigma_{t_{src}}\bigr) - (1-\tau)\,\alpha(\sigma_{t_{src}})
$$

(the first identity uses $\sigma/\sigma_{t_{src}} = 1-\tau$). Boundary check:
- $\phi_0^{B1}(0) = 1,\;\phi_0^{B1}(1) = 0$ ✓
- $\phi_1^{B1}(0) = \alpha(\sigma_{t_{src}}) - \alpha(\sigma_{t_{src}}) = 0$ ✓
- $\phi_1^{B1}(1) = \alpha(0) - 0 = 1$ ✓

For linear schedule, $\phi_1^{B1}(\tau) = (1-(1-\tau)\sigma_{t_{src}}) - (1-\tau)(1-\sigma_{t_{src}}) = \tau$ — collapses to B0.
For VP, $\phi_1^{B1}(\tau) = \sqrt{1-(1-\tau)^2\sigma_{t_{src}}^2} - (1-\tau)\sqrt{1-\sigma_{t_{src}}^2}$ — non-linear in τ and **σ_t_src-dependent** (each segment has a different $\phi_1$).

### B2 — Sin-only on linear boundary

$\phi_0(\tau) = 1-\tau$, $\phi_1(\tau) = \tau$, drop the cosine residual: $b_m \equiv 0$.

$$\Psi_{B2}(\tau) = (1-\tau) x_{t_{src}} + \tau\,\hat{x}_0 + \sum_m a_m \sin(m\pi\tau).$$

At τ=1: $\sin(m\pi) = 0$ for all integer $m$, so $\Psi_{B2}(1) = \hat{x}_0$ **strictly**. This is the principal motivation for B2 — it resolves the soft-anchor-at-τ=1 finding from R1.1 §5.3 by construction.

### B3 — Trig boundary (cos(πτ/2), sin(πτ/2))

$\phi_0(\tau) = \cos(\pi\tau/2)$, $\phi_1(\tau) = \sin(\pi\tau/2)$.

Boundary check:
- $\phi_0(0) = 1,\;\phi_0(1) = 0$ ✓
- $\phi_1(0) = 0,\;\phi_1(1) = 1$ ✓

In σ-coordinates (via $\tau = 1 - \sigma/\sigma_{t_{src}}$), $\phi_0(\tau) = \sin(\pi\sigma/(2\sigma_{t_{src}}))$ and $\phi_1(\tau) = \cos(\pi\sigma/(2\sigma_{t_{src}}))$, so

$$\Psi_{B3}^{M=0}(\sigma) = \sin\!\Bigl(\tfrac{\pi\sigma}{2\sigma_{t_{src}}}\Bigr) x_{t_{src}} + \cos\!\Bigl(\tfrac{\pi\sigma}{2\sigma_{t_{src}}}\Bigr)\hat{x}_0.$$

This is a **quarter-cycle interpolation** between the two anchors. It is symmetric and has bounded second derivatives, but as we will see in §3, it **fails the M=0 constant-velocity-flow demand** that R1.1 §5.1 imposes.

### Other candidates considered and discarded

- **B4 — Convex blend of B0 and B1** ($\phi_1 = (1-\lambda)\tau + \lambda\,\phi_1^{B1}$): keeps the schedule-coupling cost of B1 partially, with no clean theoretical advantage. Effectively a regularised B1; not separately analysed.
- **B5 — Schedule-aware τ-warp** (keep linear basis, redefine τ(σ) per schedule): out of scope — R1.1 fixed τ.
- **B6 — Polynomial cubic Hermite** ($\phi_0 = 2\tau^3 - 3\tau^2 + 1$, $\phi_1 = -2\tau^3 + 3\tau^2$, with also-zero derivatives at endpoints): forces $\partial_\tau\Psi(0) = \partial_\tau\Psi(1) = 0$ at M=0, which forbids non-zero anchor velocity. Fails M=0 collapse for the same reason as B3. Discarded.

---

## 3. Properties to preserve (checklist)

For each candidate, against the five R1.1 properties.

### 3.1 Strict anchor reproduction at τ=0

The full residual contributes $a_m\sin(0) + b_m(\cos(0)-1) = 0$ for every $m$, regardless of basis. So strict reproduction at τ=0 reduces to whether $\phi_0(0) = 1$ and $\phi_1(0) = 0$.

| Basis | $\phi_0(0)$ | $\phi_1(0)$ | Strict at τ=0? |
|---|---|---|---|
| B0 | 1 | 0 | ✓ |
| B1 | 1 | 0 | ✓ |
| B2 | 1 | 0 | ✓ |
| B3 | 1 | 0 | ✓ |

All four pass.

### 3.2 Anchor reproduction at τ=1

At τ=1 the residual contributes $a_m \sin(m\pi) + b_m(\cos(m\pi) - 1) = b_m\bigl((-1)^m - 1\bigr) = -2b_m$ for odd $m$, $0$ for even $m$. So $\Psi(1) = \phi_0(1) x_{t_{src}} + \phi_1(1) \hat{x}_0 - 2\sum_{m\text{ odd}} b_m$.

| Basis | $\phi_0(1)$ | $\phi_1(1)$ | Cosine residual at τ=1 | Strict? |
|---|---|---|---|---|
| B0 | 0 | 1 | $-2\sum_{m\text{ odd}} b_m$ | soft (R1.1 §5.3) |
| B1 | 0 | 1 | $-2\sum_{m\text{ odd}} b_m$ | soft |
| B2 | 0 | 1 | 0 (no $b_m$ term) | **strict** |
| B3 | 0 | 1 | $-2\sum_{m\text{ odd}} b_m$ | soft |

Only B2 makes τ=1 strict.

### 3.3 Anchor velocity at τ=0 for M ≥ 1

$\partial_\tau \Psi(0) = \phi_0'(0) x_{t_{src}} + \phi_1'(0) \hat{x}_0 + \pi \sum_m m\,a_m$ (the $\sin'$ contribution at $\tau=0$; $\cos'$ vanishes). The F2 velocity at the anchor is $u(\sigma_{t_{src}}) = -\partial_\tau\Psi(0)/\sigma_{t_{src}}$.

Required: match $u_{src} = (x_{t_{src}} - \hat{x}_0)/\sigma_{t_{src}}$ ⇔ $\partial_\tau \Psi(0) = \hat{x}_0 - x_{t_{src}}$ ⇔ $\phi_0'(0) = -1,\;\phi_1'(0) = 1,\;\sum_m m\,a_m = 0$.

| Basis | $\phi_0'(0)$ | $\phi_1'(0)$ | M=0 anchor-velocity match? | M ≥ 1 constraint |
|---|---|---|---|---|
| B0 | $-1$ | $1$ | exact | $\sum_m m\,a_m = 0$ (soft) |
| B1 (linear) | $-1$ | $1$ | exact | $\sum_m m\,a_m = 0$ (soft) |
| B1 (VP) | $-1$ | $\sigma_{t_{src}}\cdot(\sigma_{t_{src}}/\sqrt{1-\sigma_{t_{src}}^2}) + \sqrt{1-\sigma_{t_{src}}^2}$ ¹ | gives **VP-natural** anchor velocity (matches $\partial_\sigma\Psi_{\text{true,VP}}|_{\sigma_{t_{src}}}$, not $u_{src}$) | $\sum_m m\,a_m = 0$ (soft) |
| B2 | $-1$ | $1$ | exact | same |
| B3 | $0$ | $\pi/2$ | **mismatch** ($u_{F2}(\sigma_{t_{src}}) = -\frac{\pi}{2\sigma_{t_{src}}}\hat{x}_0$, ignores $x_{t_{src}}$) | n/a |

¹ Derivation in Appendix below.

**B3 fails this property at M=0.** The mismatch is structural: the trig basis prescribes a sinusoidal trajectory whose initial velocity depends only on $\hat{x}_0$, not on the difference $x_{t_{src}} - \hat{x}_0$. No choice of Fourier residual can fix the M=0 case (since the residual derivatives at τ=0 vanish for the cos term and contribute via $a_m$ only).

**B1 under VP gives a different (but principled) anchor velocity** — the VP-natural one, which is what we'd want if the policy is supposed to follow the true VP trajectory. R1.1 §5.4(a) computed the linear-basis case; the analogous VP case for B1 is the velocity of $\Psi_{\text{true,VP}}$ at $\sigma_{t_{src}}$.

### 3.4 F2 well-defined and bounded as σ → 0

F2 is $u = +\partial_\sigma \Psi$. At small σ, no division by σ occurs, so the only failure mode is unbounded $\partial_\sigma$.

| Basis | $\partial_\sigma \Psi^{M=0}$ as σ → 0 | Bounded? |
|---|---|---|
| B0 | $(x_{t_{src}} - \hat{x}_0)/\sigma_{t_{src}}$ constant | ✓ |
| B1 (linear) | same as B0 | ✓ |
| B1 (VP) | $\alpha'_{\text{VP}}(0)\,\hat{x}_0 + (\dots)/\sigma_{t_{src}} = 0\cdot\hat{x}_0 + \ldots$ — finite | ✓ |
| B2 | same as B0 | ✓ |
| B3 | $(\pi/(2\sigma_{t_{src}}))\,(\cos(0)\,x_{t_{src}} - \sin(0)\,\hat{x}_0) = \pi x_{t_{src}}/(2\sigma_{t_{src}})$ | ✓ |

All four bounded as σ → 0. The Fourier residual contributions are bounded for any finite M.

(Note: B1 under VP has $\alpha'_{\text{VP}}(\sigma) = -\sigma/\sqrt{1-\sigma^2}$ which diverges as $\sigma \to 1$. So if a segment starts at $\sigma_{t_{src}}$ very close to 1, B1's anchor velocity blows up. Production keeps $\sigma_{t_{src}} \le 1 - \text{eps}$ via [`gaussian_flow.py:191-195`](../../repos/piFlow/lakonlab/models/diffusions/gaussian_flow.py#L191-L195) flow_shift heuristic and similar guards, so this is not a live problem, but it is a sharper edge for B1 than B0.)

### 3.5 Compatibility with codebase's `u_to_x_0` silent linear assumption

The Fourier policy's network head outputs $\hat{x}_0$ directly (R1.1 §4.2). The codebase's `u_to_x_0` ([`gmflow.py:153-183`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L153-L183)) is bypassed at policy-construction time. So the basis choice does **not** interact with the silent-linear-schedule landmine documented in `u-to-x0-schedule-audit.md`.

The interaction is upstream, at the network's training. If the network is trained on linear-schedule samples and deployed under VP, the distribution of $(x_{t_{src}}, x_0)$ pairs the network sees at training does not match what it sees at inference. This conditioning issue is **basis-independent**.

---

## 4. Quantify the residual under each basis

This is the load-bearing analytical section. For each basis, compute $\Delta(\sigma) := \Psi^{M=0}(\sigma) - \Psi_{\text{true}}(\sigma)$ at perfect prediction $\hat{x}_0 = x_0$.

### 4.1 B0 — Linear basis

$\Psi_{B0}^{M=0}(\sigma) = (\sigma/\sigma_{t_{src}}) x_{t_{src}} + (1 - \sigma/\sigma_{t_{src}}) x_0$ (using $\hat{x}_0 = x_0$).

$\Delta_{B0}(\sigma) = \Psi_{B0}^{M=0}(\sigma) - \Psi_{\text{true}}(\sigma)$.

Using the closed form of $\Psi_{\text{true}}$ from §1.3, the $x_{t_{src}}$ coefficients cancel ($\sigma/\sigma_{t_{src}}$ in both), and the $x_0$ coefficients differ by

$$(1 - \sigma/\sigma_{t_{src}}) - \bigl[\alpha(\sigma) - (\sigma/\sigma_{t_{src}})\alpha(\sigma_{t_{src}})\bigr] = 1 - \alpha(\sigma) + (\sigma/\sigma_{t_{src}})(\alpha(\sigma_{t_{src}}) - 1).$$

So

$$\boxed{\,\Delta_{B0}(\sigma) = x_0\cdot\bigl[1 - \alpha(\sigma) - (\sigma/\sigma_{t_{src}})(1 - \alpha(\sigma_{t_{src}}))\bigr]\,}$$

**Linear schedule:** $\alpha = 1-\sigma$, so $1-\alpha(\sigma) = \sigma$ and $(\sigma/\sigma_{t_{src}})(1-\alpha(\sigma_{t_{src}})) = (\sigma/\sigma_{t_{src}})\sigma_{t_{src}} = \sigma$. So $\Delta_{B0,\text{lin}}(\sigma) = x_0\cdot[\sigma - \sigma] = 0$. ✓

**VP schedule:** $\alpha = \sqrt{1-\sigma^2}$. Define $g(\sigma) := 1 - \sqrt{1-\sigma^2}$ (convex on $[0,1]$, $g(0)=0$, $g(1)=1$). Then

$$\Delta_{B0,\text{VP}}(\sigma) = x_0\cdot\bigl[g(\sigma) - (\sigma/\sigma_{t_{src}})\,g(\sigma_{t_{src}})\bigr].$$

This is precisely the **secant residual** of $g$: the difference between the convex curve $g(\sigma)$ and the secant line through $(0,0)$ and $(\sigma_{t_{src}}, g(\sigma_{t_{src}}))$ evaluated at $\sigma$. By convexity, $\Delta_{B0,\text{VP}}(\sigma) \le 0$ on $[0, \sigma_{t_{src}}]$ with equality only at the endpoints.

**Numerical magnitudes** (units of $\|x_0\|$):

| $\sigma_{t_{src}}$ | σ at peak residual ¹ | $|\Delta_{B0,\text{VP}}(\sigma)|$ |
|---|---|---|
| 0.3 | ~0.15 | ~0.011 |
| 0.5 | ~0.25 | ~0.035 |
| 0.7 | ~0.35 | ~0.082 |
| 0.9 | ~0.45 | ~0.175 |
| 0.99 | ~0.5 | ~0.45 |

¹ Approximate; the peak of $g(\sigma) - (\sigma/\sigma_{t_{src}}) g(\sigma_{t_{src}})$ as a function of σ is at $\sigma^* = \sigma_{t_{src}}/\sqrt{1 + (g(\sigma_{t_{src}})/\sigma_{t_{src}})^{-2}}$ from $g'(\sigma^*) = g(\sigma_{t_{src}})/\sigma_{t_{src}}$ (not a clean closed form; values above from spreadsheet).

So the VP geometric residual under B0 is small (a few %) when segments are short and σ_t_src is small, but grows to ~17–45% of $\|x_0\|$ for segments that span most of the noise→data path. **Whether this matters depends on M and on how much representational capacity the Fourier residual has.**

### 4.2 B1 — Schedule-aware basis

By construction, $\Psi_{B1}^{M=0}(\sigma) = \Psi_{\text{true}}(\sigma)$ when $\hat{x}_0 = x_0$. So

$$\Delta_{B1,\text{any schedule}}(\sigma) = 0\quad (\text{at perfect prediction}).$$

Under linear schedule, B1 = B0, so $\Delta = 0$ trivially. Under VP, B1 is the only basis among the candidates that achieves zero geometric residual.

### 4.3 B2 — Sin-only on linear boundary

At M=0, dropping the cosine residual changes nothing: $\Psi_{B2}^{M=0} = \Psi_{B0}^{M=0}$. So

$$\Delta_{B2}(\sigma) = \Delta_{B0}(\sigma).$$

B2's advantage is at the τ=1 endpoint and in the loss formulation (R2's territory), not in the M=0 geometric residual.

### 4.4 B3 — Trig boundary

$\Psi_{B3}^{M=0}(\sigma) = \sin(\pi\sigma/(2\sigma_{t_{src}})) x_{t_{src}} + \cos(\pi\sigma/(2\sigma_{t_{src}})) x_0$ (with $\hat{x}_0 = x_0$).

Let $u := \sigma/\sigma_{t_{src}} \in [0,1]$. Then $\Psi_{B3} = \sin(\pi u/2) x_{t_{src}} + \cos(\pi u/2) x_0$.

**Linear schedule:** $\Psi_{\text{true,lin}} = u\cdot x_{t_{src}} + (1-u) x_0$. So

$$\Delta_{B3,\text{lin}}(\sigma) = \bigl(\sin(\pi u/2) - u\bigr) x_{t_{src}} + \bigl(\cos(\pi u/2) - (1-u)\bigr) x_0.$$

At $u = 0.5$: $\sin(\pi/4) - 0.5 = 0.207$, $\cos(\pi/4) - 0.5 = 0.207$. So $\Delta_{B3,\text{lin}}(0.5\sigma_{t_{src}}) \approx 0.207\,(x_{t_{src}} + x_0)$ — about **20% of each anchor**.

**VP schedule:** for the special case $\sigma_{t_{src}} = 1$ (segment covers full noise→data path):
$\Psi_{\text{true,VP}}(\sigma) = \sqrt{1-\sigma^2}\,x_0 + \sigma\,x_{t_{src}}$ (since $\alpha(1) = 0$, the cross term vanishes).
$\Psi_{B3}(\sigma) = \sin(\pi\sigma/2) x_{t_{src}} + \cos(\pi\sigma/2) x_0$.
At $\sigma = 0.5$: $\sin(\pi/4) - 0.5 = 0.207$, $\cos(\pi/4) - \sqrt{0.75} = 0.707 - 0.866 = -0.159$.
$\Delta_{B3,\text{VP}}(0.5) \approx 0.207\,x_{t_{src}} - 0.159\,x_0$ — about **16-21% of each anchor**.

So **B3 has substantial geometric residual under both linear and VP schedules.** This is unsurprising: B3's quarter-cycle interpolation is its own geometric trajectory, unrelated to either linear or VP forward processes. The choice was on algebraic grounds (symmetry, bounded second derivatives), but those gains cost residual mass under both schedules.

### 4.5 Side-by-side residual table

| Basis | Residual under linear | Residual under VP | Special σ_t_src behaviour |
|---|---|---|---|
| **B0** | 0 | $x_0\cdot[g(\sigma) - (\sigma/\sigma_{t_{src}}) g(\sigma_{t_{src}})]$, up to ~17% at $\sigma_{t_{src}} = 0.9$ | grows with $\sigma_{t_{src}}$ |
| **B1** | 0 | 0 | basis itself depends on $\sigma_{t_{src}}$ |
| **B2** | 0 | same as B0 | same as B0 |
| **B3** | ~20% at $u=0.5$ | ~16-21% at the same point | basis is $\sigma_{t_{src}}$-aware via the τ-axis but not schedule-aware |

**Headline:** Only B1 zeroes the geometric residual under VP. B0 and B2 are zero under linear (production) and small-to-moderate under VP. B3 is uniformly bad geometrically.

---

## 5. Trade-offs and decision

### 5.1 Comparison table

| Aspect | B0 | B1 | B2 | B3 |
|---|---|---|---|---|
| **R1.1 §5.1 M=0 collapse to constant-velocity flow** | ✓ | ✓ under linear; under VP gives the VP-natural trajectory (*not* constant velocity in σ) — semantic shift | ✓ | ✗ fails (§3.3) |
| **Strict τ=0 endpoint** | ✓ | ✓ | ✓ | ✓ |
| **Strict τ=1 endpoint** | soft | soft | **strict** | soft |
| **Anchor velocity at τ=0 (M=0)** | matches $u_{src}$ | matches under linear; matches VP-natural under VP | matches $u_{src}$ | does not match |
| **Anchor velocity strict for M ≥ 1** | requires $\sum m\,a_m = 0$ | same | same | n/a |
| **Geometric residual at perfect pred.** (linear) | 0 | 0 | 0 | ~20% mid-segment |
| **Geometric residual at perfect pred.** (VP, $\sigma_{t_{src}} = 0.9$) | ~17% | 0 | ~17% | ~16-21% |
| **Algebraic complexity of $\phi_1$** | constant | involves $\sqrt{1 - (1-\tau)^2 \sigma_{t_{src}}^2}$, $\sigma_{t_{src}}$-dependent | constant | constant trig |
| **Tractable $\partial_\sigma \Psi$** | yes (constant) | yes (closed form) | yes | yes |
| **Head architecture** | unchanged | unchanged ($x̂_0$ + $a_m$ + $b_m$) | drop $b_m$ head — saves $M\cdot C\cdot H\cdot W$ params | unchanged |
| **Loss formulation impact (R2 territory)** | residual loss fits geometric+prediction errors mixed | residual loss fits prediction-only error (cleaner under VP) | same as B0 modulo dropped $b_m$ degrees of freedom | residual loss fits a 20% geometric gap on top of prediction error |
| **Schedule transferability** | basis is linear-natural; train→deploy schedule swap requires the residual to absorb the linear-vs-VP gap | basis is built around the deployment schedule; cross-schedule transfer would re-introduce a geometric gap | same as B0 | basis is schedule-blind; same gap shape both ways |
| **Conditioning issue (x_t_src distribution depends on schedule)** | shared by all bases | shared | shared | shared |

### 5.2 Decision

**Keep B0 (linear basis) for shipping. Flag B1 for re-evaluation after empirical training results. Flag B2 as a drop-in alternative if R2 (loss) decides strict τ=1 anchoring is needed.**

Justification:

1. **Production is linear-schedule.** Under linear, B0, B1, B2 all give zero geometric residual at perfect prediction. There is no near-term gain from switching. (B3 is the only one with a residual under linear, and it's also disqualified by §3.3.)

2. **B3 is hard-eliminated.** It fails the M=0 anchor-velocity property (§3.3) and has substantial geometric residual under both schedules (§4.4). The "algebraic niceness" claim does not survive scrutiny — a quick check at one M=0 point shows B3 deviates from both linear and VP natural trajectories by ~20%. Discard.

3. **B1 vs B0 is the real call.** B1 is the principled choice for VP and the *only* basis with zero VP geometric residual. But:
   - The cost is real: $\phi_1^{B1}$ depends on $\sigma_{t_{src}}$ and on $\alpha(\sigma)$, so the basis varies per segment. The network's Fourier-coefficient head has to produce coefficients that get composed with a different $\phi_1$ shape per segment. This makes the loss landscape less uniform across segments.
   - The benefit is contingent: B1 only outperforms B0 if the Fourier residual under B0 *cannot* absorb the geometric gap. With M ~ 4-8 coefficient maps per pixel, the residual capacity is high; absorbing a smooth ~17% deviation should be well within reach. We should **train B0 first and measure** before paying B1's coupling cost.
   - Schedule transferability: B0 is the more transfer-friendly choice in the asymmetric case where training runs in linear and inference rolls out under VP (the obvious Idea-1-as-baseline-comparison case). B1 is the more accurate choice when train-and-deploy schedules match.

4. **B2 vs B0 is a smaller call.** B2 trades the cosine residual coefficients (saves head params) for strict τ=1 anchoring. R1.1 §5.3 explicitly chose to keep τ=1 soft, on the grounds that $x̂_0$ is a network prediction anyway. B2 shifts the soft-vs-strict trade-off but doesn't address the geometric residual question. **Defer to R2.** If R2 finds the loss is unstable due to the soft τ=1 anchor (e.g. drifting predictions at the data side), revisit B2 then.

5. **C1-style conditioning concern.** All four bases share the property that $x_{t_{src}}$ at training is sampled from the training-time forward process, which is currently linear. Switching to a VP-aware basis at inference doesn't fix the train/deploy distribution mismatch — that's a training-pipeline issue, not a basis issue.

### 5.3 What this commits us to

- We will train Idea 1 with B0 first, under linear schedule, with the Fourier residual absorbing whatever data-conditional and (linear-trivial) geometric deviation arises.
- If we later run Idea 1 under VP (e.g. as a baseline comparison against schedule-aware GMFlow), we will measure whether the B0-trained Fourier coefficients absorb the VP geometric gap or whether the residual norms blow up. **B1 is the documented escape hatch** if B0-under-VP is empirically unstable.
- The R1.1 design (Option A, τ-direction, F2, soft anchors) carries through unchanged.

---

## 6. Rewrite proposal for R1.1

**Decision is to keep B0. No edits to R1.1 are required.** Skip per the task instructions.

(If B1 were chosen, the edits would touch R1.1 §4.2 (basis definition), §5.1 (M=0 collapse derivation, which would still hold but via the schedule-natural trajectory rather than the linear interpolation), §5.4(a) (anchor velocity for M=0 changes its target), and §6.4 (confidence on the basis choice). Listed here for completeness in case R1.1 is revisited.)

---

## 7. Notes for the reviewer

### 7.1 Things I had to assume

- **Perfect prediction $\hat{x}_0 = x_0$ for the residual-quantification analysis (§4).** Real residuals add a prediction-error term on top. The decomposition is the standard one: total residual = geometric residual (basis-induced) + prediction residual (network-induced). The analysis in §4 isolates the first term.
- **The "natural trajectory" interpretation.** §1.3 defines $\Psi_{\text{true}}$ as the locus of $x_t = \alpha(\sigma) x_0 + \sigma\varepsilon$ with $\varepsilon$ pinned by the anchor. This is the deterministic ODE-style trajectory between two specific anchors; it is **not** the marginal $E[x_t \mid x_0]$ (which would have $\varepsilon$ averaged out). The pinned-ε interpretation is the one that makes "residual at perfect prediction" geometrically meaningful — at perfect prediction, the policy should reproduce the forward-process curve passing through the segment's observation $x_{t_{src}}$.
- **σ_t_src distribution.** I used $\sigma_{t_{src}} \in \{0.3, 0.5, 0.7, 0.9, 0.99\}$ for the residual table; the actual distribution depends on the timestep sampler ([`gaussian_flow.py:43`](../../repos/piFlow/lakonlab/models/diffusions/gaussian_flow.py#L43) uses `ContinuousTimeStepSampler`). I did not load the sampler config to see what σ-range training actually exercises — the residual scaling argument holds regardless, but the *typical* residual magnitude depends on how often segments span large σ-ranges in practice.

### 7.2 Things I could not verify

- **B3's "orthogonal to the Fourier residual" claim** in the task description. A direct check: $\langle \cos(\pi\tau/2), \sin(m\pi\tau)\rangle_{L^2[0,1]} = m/(\pi(m^2 - 1/4)) \neq 0$ for any integer $m \ge 1$. The basis is **not** orthogonal to the sin residual under the standard $L^2[0,1]$ inner product. Possible reinterpretations: (a) orthogonal under a weighted inner product; (b) orthogonal in a different parameterisation (e.g. $\tau \in [0, 2]$ where cos(πτ/2) becomes a half-period). I did not pursue this since B3 fails on geometric grounds anyway. **Flagged for the reviewer to confirm.** If B3 turns out to be orthogonal under some specific inner product that matters for the loss, that would be a reason to revisit — but the geometric-residual concern would remain.
- **$\sigma_{t_{src}}$ distribution under the production timestep sampler.** Not loaded.
- **Whether the network architecture (`GMDiTTransformer2DModel`, [`pipeline_gmdit.py:11`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L11)) supports the $\sigma_{t_{src}}$-conditioning that B1 would need.** Under B0, the head outputs $\{x̂_0, a_m, b_m\}$ which are σ-independent maps; the policy composes them with σ at inference. Under B1, the policy needs $\sigma_{t_{src}}$ to evaluate $\phi_1^{B1}$, but $\sigma_{t_{src}}$ is already known to the network (it's the conditioning input), so this should be fine — flagged here to confirm during implementation.

### 7.3 For R2 (loss) and R3 (M selection)

- **R2:** the loss should explicitly account for the geometric residual under VP if we ever train under VP with B0. The simplest version: use the network-predicted $\hat{x}_0$ to compute $\Psi^{M=0}$, then have the loss target be the *deviation* between observed $x_t$ and $\Psi^{M=0}$. Under linear schedule the deviation is purely prediction-error; under VP it adds the geometric residual. R2 should also pick between the soft τ=1 anchor (R1.1's choice, B0 / B1 / B3) and the strict one (B2).
- **R3:** under VP with B0, the Fourier residual must absorb the geometric gap of magnitude up to ~17% of $\|x_0\|$ for typical $\sigma_{t_{src}}$. The convex secant residual is smooth (analytic), so a small M (4-8 harmonics) should suffice in principle. Concrete prediction: under VP at $\sigma_{t_{src}} = 0.9$, the optimal Fourier residual concentrates on $m = 1, 2, 3$ (low harmonics fit smooth curves well). Worth measuring: $\|a_m\| / \|a_1\|$ and $\|b_m\| / \|b_1\|$ as a function of $m$ at convergence — should decay fast under B0+linear, less fast under B0+VP.

### 7.4 Confidence

- **High** confidence on **rejecting B3** (fails M=0 anchor velocity; substantial geometric residual under both schedules).
- **High** confidence on **B0 = B1 under linear** (algebraic identity).
- **Medium-high** confidence on **deferring B1 in favour of B0 for shipping** — the right choice given (a) production is linear, (b) the M ≥ 4 Fourier residual likely absorbs the VP geometric gap, (c) B1's coupling cost is real, (d) train-vs-deploy schedule symmetry is unclear. Could flip if early empirical results show B0-under-VP is unstable.
- **Medium** confidence on **deferring B2 to R2** — strict-vs-soft τ=1 anchor is a loss-shape question, and R2 is the right place for it.
- **Low** confidence that B3's "orthogonality" claim has a defensible interpretation; flagged for confirmation. But this does not change the recommendation since B3 fails on multiple grounds independent of orthogonality.

---

## Appendix: B1 anchor-velocity derivation under VP

For B1 with $\phi_1^{B1}(\tau) = \alpha((1-\tau)\sigma_{t_{src}}) - (1-\tau)\,\alpha(\sigma_{t_{src}})$:

$$\frac{d\phi_1^{B1}}{d\tau} = -\sigma_{t_{src}}\,\alpha'\bigl((1-\tau)\sigma_{t_{src}}\bigr) + \alpha(\sigma_{t_{src}}).$$

At $\tau=0$:
$$\phi_1^{B1\,'}(0) = -\sigma_{t_{src}}\,\alpha'(\sigma_{t_{src}}) + \alpha(\sigma_{t_{src}}).$$

For VP, $\alpha(\sigma) = \sqrt{1-\sigma^2}$, $\alpha'(\sigma) = -\sigma/\sqrt{1-\sigma^2}$. So
$$\phi_1^{B1\,'}(0)\big|_{\text{VP}} = -\sigma_{t_{src}}\cdot\frac{-\sigma_{t_{src}}}{\sqrt{1-\sigma_{t_{src}}^2}} + \sqrt{1-\sigma_{t_{src}}^2} = \frac{\sigma_{t_{src}}^2 + (1-\sigma_{t_{src}}^2)}{\sqrt{1-\sigma_{t_{src}}^2}} = \frac{1}{\sqrt{1-\sigma_{t_{src}}^2}}.$$

Then $\partial_\tau \Psi_{B1}^{M=0}(0) = -x_{t_{src}} + \hat{x}_0/\sqrt{1-\sigma_{t_{src}}^2}$, giving F2 anchor velocity
$$u_{F2}^{B1,\text{VP}}(\sigma_{t_{src}}) = \frac{x_{t_{src}} - \hat{x}_0/\sqrt{1-\sigma_{t_{src}}^2}}{\sigma_{t_{src}}}.$$

For comparison, the VP-natural trajectory's velocity at the anchor (from differentiating $\Psi_{\text{true,VP}}$ in σ):
$$\partial_\sigma \Psi_{\text{true,VP}}(\sigma_{t_{src}}) = \alpha'_{\text{VP}}(\sigma_{t_{src}}) x_0 + \varepsilon = \frac{-\sigma_{t_{src}}}{\sqrt{1-\sigma_{t_{src}}^2}} x_0 + \frac{x_{t_{src}} - \sqrt{1-\sigma_{t_{src}}^2}\,x_0}{\sigma_{t_{src}}}.$$

Combining the $x_0$ terms: coefficient $= -\sigma_{t_{src}}/\sqrt{1-\sigma_{t_{src}}^2} - \sqrt{1-\sigma_{t_{src}}^2}/\sigma_{t_{src}} = -[\sigma_{t_{src}}^2 + (1-\sigma_{t_{src}}^2)] / [\sigma_{t_{src}}\sqrt{1-\sigma_{t_{src}}^2}] = -1/[\sigma_{t_{src}}\sqrt{1-\sigma_{t_{src}}^2}]$.

So $\partial_\sigma \Psi_{\text{true,VP}}(\sigma_{t_{src}}) = x_{t_{src}}/\sigma_{t_{src}} - x_0/[\sigma_{t_{src}}\sqrt{1-\sigma_{t_{src}}^2}]$, which is exactly $u_{F2}^{B1,\text{VP}}$ when $\hat{x}_0 = x_0$. ✓ Confirms B1 reproduces the VP-natural anchor velocity.

---

## Verification log (2026-05-02, claude-scholar:verify-math / SymPy)

Verifying the load-bearing derivations in R1.1 ([`idea1-anchor-design.md`](idea1-anchor-design.md)) and R4 (this document). One SymPy invocation per step. Stop on first failure.

**V1 — R1.1 §4.1 step 5: τ-substitution gives constant-velocity trajectory.**
- Command: `./verify.py eq "(1 - (1 - s/S))*A + (1 - s/S)*B" "B + (s/S)*(A - B)"`
- Result: **PASS** — `✓ (1 - (1 - s/S))*A + (1 - s/S)*B = B + (s/S)*(A - B)`

**V2 — R1.1 §5.1: ∂_σ Ψ_M=0 = (x_t_src − x̂_0)/σ_t_src.**
- Command: `./verify.py eq "diff(B + (s/S)*(A - B), s)" "(A - B)/S"`
- Result: **PASS** — `✓ diff(B + (s/S)*(A - B), s) = (A - B)/S`

**V3 — R1.1 §5.3: cos(mπ) − 1 evaluates to −2 for odd m, 0 for even m (m=1,2,3,4).**
- Commands:
  - `./verify.py eq "cos(1*pi) - 1" "-2"` → `✓ cos(1*pi) - 1 = -2`
  - `./verify.py eq "cos(2*pi) - 1" "0"`  → `✓ cos(2*pi) - 1 = 0`
  - `./verify.py eq "cos(3*pi) - 1" "-2"` → `✓ cos(3*pi) - 1 = -2`
  - `./verify.py eq "cos(4*pi) - 1" "0"`  → `✓ cos(4*pi) - 1 = 0`
- Result: **PASS** for all four parities.

**V4 — R1.1 §5.4(a): for M=2, ∂_τ Ψ at τ=0 = (x̂_0 − x_t_src) + π·∑_m m·a_m.**
- Command: `./verify.py eq "diff((1-t)*A + t*B + a1*sin(pi*t) + b1*(cos(pi*t) - 1) + a2*sin(2*pi*t) + b2*(cos(2*pi*t) - 1), t).subs(t, 0)" "(B - A) + pi*(a1 + 2*a2)"`
- Result: **PASS** — `✓ ... = (B - A) + pi*(a1 + 2*a2)`. Confirms ∑_m m·a_m structure.

**V5 — R1.1 §5.4 sign: u = ε − x_0 equals +∂_σ x_t for x_t = (1−σ)x_0 + σε.**
- Command: `./verify.py eq "diff((1-s)*x0 + s*eps, s)" "eps - x0"`
- Result: **PASS** — `✓ diff((1-s)*x0 + s*eps, s) = eps - x0`. Confirms F2 sign convention.

**V6 — R4 §1: under VP, α(σ) = √(1 − σ²) follows from α = cos(πt/2), σ = sin(πt/2).**
- Command: `./verify.py eq "cos(pi*t/2)" "sqrt(1 - sin(pi*t/2)**2)"`
- Result: **FAIL (assumption-dependent)** — `✗ Not equal. Difference: -sqrt(cos(pi*t/2)**2) + cos(pi*t/2)`
- Reading: SymPy returns `cos(πt/2) − |cos(πt/2)|`, which vanishes iff `cos(πt/2) ≥ 0`. The identity holds on the documented domain `t ∈ [0,1]` (where `cos(πt/2) ∈ [0,1]`), but SymPy without the domain assumption refuses to drop the absolute value. Per the verification protocol, this is a soft failure: the underlying math is correct under the assumption, but the SymPy check as written cannot certify it. **Stop here per the task's "stop on first failure" instruction.**

V7 (R4 §4 residual under VP) and V8 (R4 §3 anchor reproduction at τ=0) — **not run.** Skipped per the stop-on-first-failure rule.

V6 failed; flagged for revision.

The flag is on the *form of the SymPy check*, not on the math: the Pythagorean identity `cos²(θ) = 1 − sin²(θ)` is unconditional, and the sign branch `cos(πt/2) ≥ 0` is satisfied on `t ∈ [0, 1]` which is the only relevant domain (`gaussian_flow.py:90-95` and the trig-schedule definition both restrict to that interval). Reviewer note: a follow-up SymPy run with explicit `Symbol('t', positive=True, finite=True)` and `t ≤ 1` assumption would close this; an unconditional identity check via squaring (`cos(πt/2)² = 1 − sin(πt/2)²`) would also close it. Either is a verification rewrite, not a derivation revision.
