# Idea 1 — Fourier Policy: Endpoint Anchor Design Decision

**Date:** 2026-05-02 (R1.1 — supersedes R1 of 2026-05-01)
**Scope:** Decide what `X_start(ξ)` and `X_end(ξ)` should be in the Fourier-policy formula

$$\Psi_\Pi(\tau,\xi) = \phi_0(\tau)\,X_{start}(\xi) + \phi_1(\tau)\,X_{end}(\xi) + \sum_{m=1}^{M}\bigl(a_m(\xi)\sin(m\pi\tau) + b_m(\xi)(\cos(m\pi\tau)-1)\bigr)$$

with linear basis $\phi_0(\tau) = 1-\tau$, $\phi_1(\tau) = \tau$ ([personal-docs/pi_flow_three_ideas.md:37](../three-ideas.md), [personal-docs/worknotes/ideas-comparison.md:25](../ideas-comparison.md)).

This is a structural choice with consequences for: (1) head architecture, (2) training procedure, (3) inference procedure, (4) loss formulation, (5) compatibility with PiFlow’s segment-distillation contract.

R1 surfaced two internal inconsistencies (τ-axis direction; velocity-recovery formula). This R1.1 resolves them with explicit derivations in §4, §5.1, §5.3, §5.4, and propagates the fix into §1.6, §2 Option A, §3, §6. See "Changes from prior version" at the end for a precise diff.

---

## 1. Time and Schedule Convention (factual)

### 1.1 Physical time, sigma, alpha

- `time_scaling = num_train_timesteps` (e.g. 1000) is set in [`gmflow.py:147`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L147).
- Normalised noise level is `σ = t / time_scaling` ([`gmflow.py:161`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L161); [`gmflow.py:206`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L206); [`gmflow.py:212`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L212)). So `σ ∈ [0, 1]`.
- The legacy posterior JIT hard-codes the linear/VE schedule `α = 1 - σ` ([`gmflow.py:51-52`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L51-L52)).
- The schedule-agnostic JIT (`gmflow_posterior_mean_jit_general`, [`gmflow.py:79-139`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L79-L139)) accepts arbitrary `(α, σ)` pairs and is currently routed only via the `GMFlowMixin.gmflow_posterior_mean` wrapper ([`gmflow.py:185-252`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L185-L252)).
- Wrapper status and call-site inventory: [derivations/03_rollout_plan.md:14-39](../../derivations/03_rollout_plan.md#L14-L39), [derivations/03_rollout_plan.md:65-72](../../derivations/03_rollout_plan.md#L65-L72).

### 1.2 Direction of time, forward process, and the velocity convention

- The training-time forward process is hardcoded linear/VE: `x_t = (1-σ) x_0 + σ ε` ([`gaussian_flow.py:90-95`](../../repos/piFlow/lakonlab/models/diffusions/gaussian_flow.py#L90-L95)).
- The training velocity target is `u = ε − x_0` ([`gaussian_flow.py:138`](../../repos/piFlow/lakonlab/models/diffusions/gaussian_flow.py#L138)).
- Differentiating the forward process: `∂_σ x_t = −x_0 + ε = ε − x_0 = u`. So **along the deterministic trajectory in this codebase, `u = +∂_σ x_t`** (not `−∂_σ x_t`). This sign matters for B2 below.
- Sampling direction is high-`σ` → low-`σ`. Evidence: [`pipeline_gmdit.py:80-81`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L80-L81) iterates `timestep_id` ascending while `scheduler.timesteps` are descending in `t`.
- So `σ = 1` ↔ pure-noise endpoint, `σ = 0` ↔ clean-data endpoint.

### 1.3 VP vs VE consequences

- VE (current): `α = 1 - σ`, SNR `= (1-σ)²/σ²`.
- VP (trig): `α = cos(πt/2)`, `σ = sin(πt/2)`, SNR `= cot²(πt/2)`. SNR blows up as `t → 0`; the existing `clamp(min=eps)` may be insufficient ([ideas-comparison.md:49](../ideas-comparison.md#L49)).
- The schedule-agnostic posterior covers VP at the GMM-policy level, but only on routes that go through the wrapper; the policy-class call sites in `piflow_policies/gmflow.py` are intentionally schedule-locked ([`piflow_policies/gmflow.py:8-13`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/gmflow.py#L8-L13)).

### 1.4 Policy contract (PiFlow segment distillation)

- `BasePolicy.pi(x_t, sigma_t) → u` ([`piflow_policies/base.py:7`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/base.py#L7)). The contract receives `x_t`; it does not require that `u` *depend* on `x_t`.
- The policy is constructed from one expensive forward pass at `(x_t_src, σ_t_src)` ([`piflow_policies/gmflow.py:33-47`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/gmflow.py#L33-L47); [`piflow_policies/dx.py:23-46`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L23-L46)) and is then queried cheaply at any `σ_t ≤ σ_t_src` within the segment ([`piflow_policies/gmflow.py:60-94`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/gmflow.py#L60-L94); [`piflow_policies/dx.py:77-101`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L77-L101)).
- The segment is bounded by `(t_src, t_dst)`. For `DXPolicy`, `segment_size` and `shift` are intrinsic and immutable across elastic inference ([`piflow_policies/dx.py:13-19`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L13-L19)). For `GMFlowPolicy`, no explicit segment bound is stored — only the source `(x_t_src, σ_t_src)` ([`piflow_policies/gmflow.py:40-47`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/gmflow.py#L40-L47)).
- **Anchor reproduction property**: when queried at `(x_t_src, σ_t_src)`, GMFlowPolicy returns the network’s direct GM mean ([`piflow_policies/gmflow.py:74-75`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/gmflow.py#L74-L75)) and then converts to velocity via `u = (x_t − x_0)/σ` ([`piflow_policies/gmflow.py:93`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/gmflow.py#L93)). At the anchor this gives `u_src = (x_t_src − x̂_0)/σ_t_src` exactly.
- **Both existing policies parametrise an `x_0`-estimate, not the trajectory.** DXPolicy ([`piflow_policies/dx.py:100`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L100)) and GMFlowPolicy ([`piflow_policies/gmflow.py:93`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/gmflow.py#L93)) both recover velocity as `u = (x_t − x_0(σ))/σ`. This is the F1 family in B2 below. Idea 1’s Fourier formula is structurally different — it parametrises the *trajectory* `Ψ(σ) ≈ x_t(σ)` (per [pi_flow_three_ideas.md:44-46](../three-ideas.md#L44-L46): "the velocity is … obtained by differentiating the function") — and therefore selects a different velocity-recovery formula (F2 below).

### 1.5 What is known at inference time

At each segment in [`pipeline_gmdit.py:80-138`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L80-L138):

| Quantity | Known? | Source |
|---|---|---|
| `t` (current segment top) | yes | [`pipeline_gmdit.py:81`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L81) |
| `x_t` (current state at `t`) | yes | cached as `x_t_base` at [`pipeline_gmdit.py:123`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L123) |
| Network output at `(x_t, t)` | yes (one forward pass) | [`pipeline_gmdit.py:87-91`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L87-L91) |
| Substep `t_sub < t` | yes | [`pipeline_gmdit.py:135`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L135) |
| Original noise sample `x_T` | yes (only the very first step uses it) | [`pipeline_gmdit.py:60-64`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L60-L64) |
| `x_0` (clean target) | **no** at inference; predicted by network |  — |
| Future `x_t_dst` (next segment) | **no** until sampler steps |  — |

So at policy-construction time, the cheap, exactly-known anchors are `(x_t_src, σ_t_src)` and (only globally) `x_T`. The data-side quantity `x_0` is always a network prediction.

### 1.6 What “t” means in the Fourier formula (assumption, narrowed)

The whiteboard formula uses argument `t ∈ [0,1]`. Nothing in [pi_flow_three_ideas.md:37](../three-ideas.md#L37) or [ideas-comparison.md:25](../ideas-comparison.md#L25) pins this to physical `t`, to `σ`, or to a segment-local re-parametrisation. **I assume the formula’s `t` is a segment-local re-parametrisation `τ ∈ [0,1]`**, with the τ↔σ direction fixed by the M=0 collapse derivation in §4. Under that derivation, **τ=0 ↔ σ=σ_t_src (high-noise / segment anchor); τ=1 ↔ σ=0 (clean data)**.

---

## 2. Anchor Options (in detail)

For each option I state: the meaning of `X_start`, `X_end`; the implied τ-axis; what the network must output; how segments embed; and the headline trade-offs. The τ-direction and velocity-recovery choices are derived in §4 and §5; this section uses those choices.

### Option A — Segment-local, `X_start = x_t_src` (cached), `X_end = x̂_0` (predicted)

- **τ-axis**: `τ = 0` ↔ `σ = σ_t_src` (high-noise end of segment), `τ = 1` ↔ `σ = 0` (clean data, *not* `σ_t_dst`). Concretely `τ(σ) = 1 − σ/σ_t_src`.
- **`X_start(ξ) = x_t_src(ξ)`** — already in memory; no network output needed for it.
- **`X_end(ξ) = x̂_0(ξ)`** — head outputs a single tensor of shape `(B, C, H, W)`, i.e. the predicted clean image.
- **Network outputs**: `{x̂_0}` plus `{a_m(ξ), b_m(ξ)}_{m=1..M}` (each a `(B, C, H, W)` map). `(2M+1)·C·H·W` scalars per sample.
- **Velocity (F2)**: `u(σ_t) = +∂_σ Ψ(σ_t) = −(1/σ_t_src) · ∂_τ Ψ(τ)|_{τ=1−σ_t/σ_t_src}`. The velocity is x_t-independent within a segment. See §4 and §5.4 for the derivation and consequences.
- **Anchor segments**: each segment is a sub-interval `τ ∈ [0, τ_dst]` of `[0,1]` where `τ_dst = 1 − σ_dst/σ_src ∈ (0, 1)`. The function value at `τ=1` (i.e. `σ=0`) is the data endpoint; the policy is queried only inside the segment, so we never actually evaluate at `τ=1` during a sampling pass — but the network still must place a meaningful `x̂_0` there because the segment’s shape near `τ=τ_dst` depends on it.
- **Pros**: matches Idea 1’s stated "differentiate the function" semantics ([pi_flow_three_ideas.md:44-46](../three-ideas.md#L44-L46)); strict anchor reproduction at `τ=0` for free (sin and (cos−1) both vanish there); only one extra clean-image head beyond the Fourier coefficients; degenerates cleanly to constant-velocity flow at M=0 (see §5.1).
- **Cons**: τ is a segment-dependent re-mapping of σ, so the same `{a_m, b_m}` mean different physical-σ trajectories under different `σ_t_src`. Need to fix `τ(σ)` at training time and re-state it at inference (analogous to DX’s `segment_size, shift` invariants — [`piflow_policies/dx.py:13-19`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L13-L19)). And: under F2, velocity is x_t-independent within a segment, which differs from how DX/GMFlow handle SDE deviations (§5.4).

### Option B — Global, `X_start = x_T` (the original noise sample), `X_end = x̂_0`

- **τ-axis**: `τ = 0` ↔ `σ = 1` (true noise endpoint), `τ = 1` ↔ `σ = 0`. Same axis for every segment.
- **`X_start = x_T`**: known once, at [`pipeline_gmdit.py:60-64`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L60-L64). Persists across segments.
- **`X_end = x̂_0`**: predicted at every segment by the network.
- **Velocity (F2)**: `u = +∂_σ Ψ`.
- **Pros**: globally well-defined trajectory; very natural Fourier basis on `[0,1]`; no per-segment re-parameterisation. Cheap evaluation.
- **Cons**: no segment-local anchor reproduction. Between segments, `{a_m, b_m, x̂_0}` change but `X_start` does not — `Ψ` jumps discontinuously at segment boundaries unless the loss explicitly enforces continuity. Also: the policy at the start of segment k (with state `x_t_src^(k) ≠ x_T`) does not pass through `x_t_src^(k)`, so the segment’s on-trajectory point at σ=σ_t_src^(k) is not the actual sampler state. Inelegant.

### Option C — Both endpoints predicted by the network (free anchors)

- **τ-axis**: as in Option A.
- **`X_start = x̂_{start}`** and **`X_end = x̂_{end}`** are both head outputs.
- **Network outputs**: `{x̂_{start}, x̂_{end}, a_m, b_m}` — one extra `(B, C, H, W)` map.
- **Velocity (F2)**: as in Option A.
- **Pros**: most expressive; the network can “smooth” a noisy `x_t_src` into `x̂_{start}` before differentiating, which may be numerically helpful near small `σ`.
- **Cons**: doubles the heaviest output channel (`(B,C,H,W)` ×2); needs an explicit auxiliary loss to keep `x̂_{start} ≈ x_t_src` (otherwise the trajectory is decoupled from the actual current state and the multi-segment iteration drifts). Strictly worse than Option A *unless* the auxiliary loss is well-tuned.

### Option D — Segment-local with `X_start = x_t_src`, `X_end = x̂_{t=0}` *and* a separate segment-end anchor `X_dst = x̂_{t_dst}`

- Adds a third, intermediate anchor on the segment endpoint to pin the function at `σ = σ_dst` as well.
- **Pros**: makes inter-segment continuity strict.
- **Cons**: `t_dst` is not fixed under elastic inference (you don’t know in advance how the sampler will subdivide). Either you pick `t_dst` at training time (loses elasticity) or you ignore it (defeats the point). High complexity, low payoff. Not pursued further.

### Option E — Pure Fourier, no boundary terms

- `X_start = X_end = 0`; the basis carries everything.
- **Pros**: simplest formula.
- **Cons**: strong implicit prior that the path passes through 0 at `τ=0` and `τ=1`. False for both noise and data. Either the network compensates with very large coefficients (numerical pain) or the policy is wrong at endpoints. Discarded.

### Boundary-term aside (resolved here, see §5.3)

The basis function `(cos(mπτ) − 1)` vanishes at `τ=0` for all `m` (good — strict anchor reproduction at the segment-anchor side is automatic), but at `τ=1` it equals `(−1)^m − 1 = −2` for odd `m` and `0` for even `m`. So `Ψ(τ=1) = X_end − 2·∑_{m\,\text{odd}} b_m`, **not** `X_end` strictly. Under the resolved τ-direction, this affects only the **data-side** endpoint (`σ=0`), which is a soft prediction (`X_end = x̂_0`) anyway. **Resolution adopted in this revision: accept soft anchoring of `X_end`** (option (3) from R1’s aside). Justification in §5.3.

---

## 3. Per-option Analysis Table

Legend: `+` = clear advantage, `−` = clear disadvantage, `~` = neutral / depends. Heads-of-comparison are the five consequences the question calls out.

| Axis | A. start=x_t_src, end=x̂_0 | B. start=x_T, end=x̂_0 | C. both predicted | D. + segment-end anchor | E. no boundaries |
|---|---|---|---|---|---|
| **Head architecture** (extra outputs vs status quo) | `+` one `(B,C,H,W)` head (x̂_0) + `2M·C·H·W` Fourier coeffs | `+` same as A | `−` two `(B,C,H,W)` heads + Fourier | `−` three `(B,C,H,W)` heads + Fourier | `+` only Fourier coeffs |
| **Training procedure** (loss complexity) | `+` x̂_0 has a natural target (true `x_0`); coefficients trained by trajectory loss | `+` same target structure; but residual must absorb the entire path from noise to data globally | `−` needs auxiliary loss to pin x̂_{start} to x_t_src | `−` needs anchor losses for two intermediate points | `−` no anchors → loss must shape entire path; pathological gradients |
| **Inference procedure** (compatibility with `pipeline_gmdit.py`) | `+` plugs into `pi(x_t, σ_t) → u` ([`piflow_policies/base.py:7`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/base.py#L7)) via F2 (`u = +∂_σ Ψ`); contract receives `x_t` but does not require it be used | `−` `Ψ` does not pass through current `x_t_src` at segment start → trajectory mismatch | `~` like A, with extra forward op | `−` requires storing `t_dst` per segment, breaks elastic inference | `~` like A in the engineering sense |
| **Loss formulation** (compatibility with GMFlow loss family) | `+` x̂_0 supervision is the same training signal GMFlow already uses ([`gmflow.py:439-452`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L439-L452)) | `~` x̂_0 supervised the same way; but `X_start = x_T` couples loss across segments | `−` extra term for `x̂_{start}`; competing gradients | `−` three coupled anchor losses | `−` no clean target for boundary terms |
| **Compat. w/ segment distillation contract** | `+` strict anchor reproduction at `τ=0`: `Ψ(σ_t_src) = x_t_src` for all `{a_m, b_m}` (sin and (cos−1) vanish at τ=0); elastic-inference safe if `τ(σ)` is fixed at train time | `−` no segment-local anchor reproduction | `~` works but loses the “anchor is ground truth” property | `−` `t_dst` invariance breaks elasticity | `−` no anchor at all |
| **Compat. w/ schedule-agnostic posterior infra** ([derivations/03_rollout_plan.md:14-23](../../derivations/03_rollout_plan.md#L14-L23)) | `~` does *not* need the GMM posterior — it parametrises X(σ) directly. But it does need an `(α, σ)`-aware `τ(σ)` map; the posterior infrastructure is orthogonal. | `~` same | `~` same | `~` same | `~` same |
| **Robustness under VP at small `σ`** ([ideas-comparison.md:49](../ideas-comparison.md#L49)) | `+` velocity recovered as `+∂_σ Ψ(σ)` with bounded `Ψ`; **does not divide by σ** → no `1/σ` blowup at small σ. This is a real advantage of F2 over F1-style recovery. | `~` same; but global function may need higher M to fit small-`σ` rapid changes | `~` like A | `~` like A | `−` no anchor near `σ=0`, residual must do all the work where SNR diverges |

---

## 4. Recommendation 

### 4.1 The two structural questions — derived from scratch

**Q1 (B1): which τ↔σ direction?** Demand: at M=0, the policy reduces to constant-velocity flow between `(x_t_src, σ_t_src)` and `(x̂_0, 0)`.

Derivation:

1. The constant-velocity flow between those two points has trajectory
   $$x_t(\sigma) = \hat{x}_0 + \frac{\sigma}{\sigma_{t_{src}}}\bigl(x_{t_{src}} - \hat{x}_0\bigr), \quad \sigma \in [0, \sigma_{t_{src}}].$$
   Its derivative is `∂_σ x_t = (x_t_src − x̂_0)/σ_t_src ≡ u_src`, constant.
2. The Fourier formula at `M = 0` with linear basis is
   $$\Psi(\tau) = (1-\tau)\,X_{start} + \tau\,X_{end}.$$
3. Choose `X_start = x_t_src`, `X_end = x̂_0`. Endpoint requirements:
   - `Ψ(σ = σ_t_src) = x_t_src` ⇒ `τ(σ_t_src) = 0` (so `(1−τ)` weight is on `X_start`).
   - `Ψ(σ = 0) = x̂_0` ⇒ `τ(0) = 1` (so `τ` weight is on `X_end`).
4. The unique linear map satisfying both is
   $$\boxed{\,\tau(\sigma) = 1 - \frac{\sigma}{\sigma_{t_{src}}}\,}$$
   with inverse `σ(τ) = (1 − τ)·σ_t_src`.
5. Check: substituting `τ = 1 − σ/σ_t_src` into `Ψ(τ) = (1−τ)x_t_src + τ·x̂_0` gives `Ψ(σ) = (σ/σ_t_src)·x_t_src + (1 − σ/σ_t_src)·x̂_0 = x̂_0 + (σ/σ_t_src)(x_t_src − x̂_0)`, which is the line from step 1. ✓

This matches DXPolicy with N=2 grid points in the loose sense the question requires: a deterministic linear interpolation between the segment’s top anchor and a clean-image estimate, recovering a constant `u` along the segment. (Strictly, DXPolicy with N=2 stores two `x_0` candidates and interpolates *those*, then reads `u = (x_t − x_0(σ))/σ` at [`piflow_policies/dx.py:100`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L100); see §5.2 for why the “constant-velocity flow between (x_t_src, σ_t_src) and (x̂_0, 0)” reduction holds for both DX-N=2 and Idea 1-M=0 even though the intermediate parametrisations differ.)

**Q2 (B2): F1 = `(x_t − Ψ)/σ_t` or F2 = `+∂_σ Ψ`?** See §5.4 for the full derivation of (a)/(b)/(c). Headline:

- F1 fails (a): at the anchor it returns `0`, not `u_src`.
- F1 fails (b): along the deterministic M=0 segment with `x_t = Ψ(σ)`, F1 returns `0` everywhere, not the constant `u_src`.
- F2 passes (a) and (b) by construction (it is just `∂_σ` of the trajectory).
- F1 “catches up” after SDE deviation (c); F2 does not.

**Pick F2.** F2 is the formula that makes Idea 1’s "differentiate the function" semantics ([pi_flow_three_ideas.md:44-46](../three-ideas.md#L44-L46)) consistent with the codebase’s `u = +∂_σ x_t` velocity convention (§1.2). F1 only makes sense if `Ψ` is reinterpreted as an `x_0`-estimate — which is the DX/GMFlow parametrisation, *not* Idea 1.

### 4.2 Recommended Fourier-policy specification

Pick **Option A** with the conventions derived above:

- Output of `G_θ(x_t_src, t_src)`: `{x̂_0, {a_m, b_m}_{m=1..M}}`.
- τ-axis: `τ(σ) = 1 − σ/σ_t_src`, so `τ=0` ↔ `σ=σ_t_src`, `τ=1` ↔ `σ=0`.
- Trajectory function:
  $$\Psi(\tau) = (1-\tau)\,x_{t_{src}} + \tau\,\hat{x}_0 + \sum_{m=1}^{M}\bigl(a_m\sin(m\pi\tau) + b_m(\cos(m\pi\tau)-1)\bigr).$$
- Velocity (F2):
  $$u(\sigma_t) = +\partial_\sigma \Psi(\sigma_t) = \frac{d\tau}{d\sigma}\,\partial_\tau \Psi(\tau)\Big|_{\tau=1-\sigma_t/\sigma_{t_{src}}} = -\frac{1}{\sigma_{t_{src}}}\,\partial_\tau \Psi(\tau).$$
- The velocity is x_t-independent within a segment. This is a **deliberate departure** from the F1 pattern used by DXPolicy ([`piflow_policies/dx.py:100`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L100)) and GMFlowPolicy ([`piflow_policies/gmflow.py:93`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/gmflow.py#L93)). See §5.4 (c) for the consequence under SDE-deviated x_t and §6.2 (now-resolved item) for the hybrid F2-plus-correction option deferred to a follow-up.

### 4.3 Why this passes the criteria

1. **Contract compatibility.** `pi(x_t, σ_t) → u` ([`piflow_policies/base.py:7`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/base.py#L7)) receives `x_t` but does not require `u` to depend on it. F2 is contract-legal.
2. **Anchor reproduction at `(x_t_src, σ_t_src)`.** At `τ=0`: `sin(0) = 0` and `cos(0) − 1 = 0`, so all Fourier residual terms vanish identically. `Ψ(τ=0) = X_start = x_t_src` and `u(σ_t_src) = u_src` for any `{a_m, b_m}`. **Strict, by construction, no constraint on the coefficients.**
3. **Minimal head growth.** One extra `(B, C, H, W)` map (`x̂_0`) plus `2M` coefficient maps. The `x̂_0` head is the same quantity GMFlow’s loss already supervises ([`gmflow.py:454-474`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L454-L474)).
4. **Clean degeneracy.** Collapses to constant-velocity flow at M=0 (§5.1).
5. **Schedule-agnosticism orthogonal.** Under any α(σ) schedule, the trajectory is parametrised in σ via τ(σ); the schedule choice enters only through how the sampler maps physical t to σ. Under linear (`α = 1 − σ`) or trig (`α² + σ² = 1`), the same `{x̂_0, a_m, b_m}` define the same `Ψ(σ)` — what changes is the time-discretisation the sampler picks. The schedule-agnostic posterior infrastructure ([derivations/03_rollout_plan.md:14-23](../../derivations/03_rollout_plan.md#L14-L23)) is not consumed here, but coexists cleanly.
6. **Elastic inference.** As long as `τ(σ) = 1 − σ/σ_t_src` is held fixed at training and inference (analogous to `DXPolicy.segment_size, shift` — [`piflow_policies/dx.py:13-19`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L13-L19)), changing the number of inference substeps does not change the policy’s meaning.
7. **Numerical advantage at small σ.** F2 does not divide by σ, so there is no `1/σ` blowup near the data endpoint — a structural improvement over F1-style recovery. This is the bright spot for VP, where SNR diverges at small σ ([ideas-comparison.md:49](../ideas-comparison.md#L49)).

---

## 5. Sanity / Collapse Checks for the Top Pick (Option A)

### 5.1 Collapse to no-Fourier (M = 0) (rewritten)

Set `M = 0`. The trajectory function reduces to
$$\Psi(\tau) = (1-\tau)\,x_{t_{src}} + \tau\,\hat{x}_0.$$
Substituting `τ(σ) = 1 − σ/σ_t_src`:
$$\Psi(\sigma) = \frac{\sigma}{\sigma_{t_{src}}}\,x_{t_{src}} + \left(1 - \frac{\sigma}{\sigma_{t_{src}}}\right)\hat{x}_0 = \hat{x}_0 + \frac{\sigma}{\sigma_{t_{src}}}\bigl(x_{t_{src}} - \hat{x}_0\bigr).$$

Endpoint check:
- `Ψ(σ = σ_t_src) = x_t_src` ✓ (matches the segment’s noisy anchor).
- `Ψ(σ = 0) = x̂_0` ✓ (matches the predicted clean image).

Velocity along the deterministic trajectory (F2):
$$u(\sigma) = \partial_\sigma \Psi(\sigma) = \frac{x_{t_{src}} - \hat{x}_0}{\sigma_{t_{src}}} \equiv u_{src} \quad \text{(constant in } \sigma\text{)}.$$

This is the constant-velocity flow demanded in B1. **The τ-direction `τ = 1 − σ/σ_t_src` is the unique linear choice (with `X_start = x_t_src`, `X_end = x̂_0`) that produces this collapse.** Any reversal of the direction would put `X_start` at the data side and `X_end` at the noise side, which contradicts the `(start, end)` reading order for forward generation (noise → data) used in §1.6.

R1’s §5.1 “correction” (`τ = σ/σ_t_src` with `φ_0(σ) = σ/σ_t_src` anchoring `x_t_src`) is the same trajectory equation under different labels: `σ/σ_t_src = 1 − τ` for our `τ`, so calling that quantity “τ” is a relabelling, not a different convention. R1’s confusion was notational, not mathematical. This R1.1 fixes the labelling once and uses it throughout.

### 5.2 Collapse to GMFlow (gross check) — preserved from R1

GMFlow’s policy is *not* a deterministic interpolant; it is a Bayesian posterior over a GMM ([`piflow_policies/gmflow.py:78-92`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/gmflow.py#L78-L92)). Idea 1’s Fourier policy *replaces* the GMM with a deterministic Fourier path. So Option A does **not** reduce to GMFlow even at M=0. The natural baseline is constant-velocity flow between the two endpoints, which is closer in spirit to DXPolicy ([`piflow_policies/dx.py:51-101`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L51-L101)) than to GMFlow. (Strictly: DXPolicy with N=2 grid points stores two `x_0` candidates that the network outputs at the segment endpoints, so the parametrisations differ even though they collapse to the same constant-velocity trajectory in the special case where both candidates coincide.)

### 5.3 Anchor-reproduction at `(x_t_src, σ_t_src)` (rewritten)

At the segment-anchor side (`σ = σ_t_src`, `τ = 0`):
$$\Psi(\tau = 0) = (1-0)\,X_{start} + 0\cdot X_{end} + \sum_m a_m\,\underbrace{\sin(0)}_{=0} + b_m\underbrace{(\cos(0) - 1)}_{=0} = X_{start} = x_{t_{src}}.$$

**Strict anchor reproduction holds at `τ=0` for all coefficients `{a_m, b_m}`, with no constraint required.** This is a structural property of the basis: `sin(mπ·0) = 0` and `cos(mπ·0) − 1 = 0` for every `m ≥ 1`.

At the data side (`σ = 0`, `τ = 1`):
$$\Psi(\tau = 1) = X_{end} + \sum_m a_m\,\underbrace{\sin(m\pi)}_{=0} + b_m\underbrace{(\cos(m\pi) - 1)}_{=(-1)^m - 1} = X_{end} - 2\sum_{m\,\text{odd}} b_m.$$

So **strict anchor reproduction fails at `τ=1`** unless `∑_{m odd} b_m = 0` or the basis is modified.

**Resolution adopted**: accept `X_end` as a soft anchor. Justification:
1. `X_end = x̂_0` is a network prediction in any case — it is supervised by the ground-truth `x_0` through the reconstruction loss, not constrained as a hard interpolation node.
2. The data side (`σ = 0`) is *not actually visited* by the sampler within a segment — segments end at `σ_t_dst > 0` (and the global sampling pass terminates at the smallest sampled `σ`, also typically > 0). So the soft endpoint at `τ=1` only matters via its influence on `Ψ(τ < 1)` along the segment — and that influence is what the loss will shape.
3. The cosine offset `−2·∑_{m odd} b_m` is an additive perturbation that the model can learn to absorb into the prediction `x̂_0` itself. In other words, the network can compensate by predicting `x̂_0' = x̂_0 + 2·∑_{m odd} b_m` such that `Ψ(τ=1) = x̂_0`. Whether to expose this trade-off explicitly is a training-time choice.
4. The strict-anchor alternatives (drop odd `b_m`; modify `φ_1` with a `−2τ·∑_{m odd} b_m` correction; switch to sin-only basis) all reduce representational capacity or add awkward construction terms. None gives a benefit that outweighs its cost given (1)–(3).

The strict-anchor remedies are deferred to a follow-up if empirical training shows the soft endpoint is unstable.

### 5.4 Velocity-recovery: F1 vs F2 (rewritten — full B2 derivation)

Define on Option A’s `Ψ`:
- **F1**: `u_F1(x_t, σ_t) = (x_t − Ψ(σ_t))/σ_t`.
- **F2**: `u_F2(σ_t) = +∂_σ Ψ(σ_t)`.

Sign convention: in this codebase, along the deterministic trajectory, `u = +∂_σ x_t` (derived in §1.2 from [`gaussian_flow.py:90-95`](../../repos/piFlow/lakonlab/models/diffusions/gaussian_flow.py#L90-L95) and [`gaussian_flow.py:138`](../../repos/piFlow/lakonlab/models/diffusions/gaussian_flow.py#L138)). So F2 must be `+∂_σ Ψ`, **not** `−∂_σ Ψ` (R1’s §5.4 had the wrong sign).

#### (a) Anchor (σ = σ_t_src, τ = 0, on-trajectory: x_t = x_t_src)

The network’s direct velocity prediction is `u_src = (x_t_src − x̂_0)/σ_t_src` (since the network outputs `x̂_0` and the codebase relates `u = ε − x_0`, with `x_t_src = (1 − σ_t_src) x̂_0 + σ_t_src ε ⇒ ε = (x_t_src − (1 − σ_t_src) x̂_0)/σ_t_src ⇒ u = ε − x̂_0 = (x_t_src − x̂_0)/σ_t_src`).

- **F1 at anchor**: `Ψ(σ_t_src) = Ψ(τ=0) = x_t_src` (§5.3). So `u_F1 = (x_t_src − x_t_src)/σ_t_src = 0`. **Does NOT match `u_src`.**
- **F2 at anchor**: For M=0, `∂_σ Ψ = (x_t_src − x̂_0)/σ_t_src = u_src`. For M≥1, `∂_σ Ψ = u_src + (1/σ_t_src)·∑_m [m π · a_m cos(m π τ) + m π · b_m sin(m π τ)]·(−1)`. Wait — let me redo this carefully.

For M ≥ 1:
$$\partial_\tau \Psi(\tau) = -x_{t_{src}} + \hat{x}_0 + \sum_m \bigl(a_m\, m\pi\cos(m\pi\tau) - b_m\, m\pi\sin(m\pi\tau)\bigr).$$
$$\partial_\sigma \Psi(\sigma) = \partial_\tau \Psi \cdot \frac{d\tau}{d\sigma} = -\frac{1}{\sigma_{t_{src}}}\,\partial_\tau \Psi(\tau).$$

At τ=0:
$$\partial_\tau \Psi(0) = -x_{t_{src}} + \hat{x}_0 + \sum_m a_m\, m\pi\cdot 1 + 0 = (\hat{x}_0 - x_{t_{src}}) + \pi\sum_m m\,a_m.$$
$$u_{F2}(\sigma_{t_{src}}) = -\frac{1}{\sigma_{t_{src}}}\bigl[(\hat{x}_0 - x_{t_{src}}) + \pi\sum_m m\,a_m\bigr] = u_{src} - \frac{\pi}{\sigma_{t_{src}}}\sum_m m\,a_m.$$

So F2 at the anchor matches `u_src` exactly when `∑_m m·a_m = 0`. This is a **soft** match for general M: the model can either learn `a_m` such that the sum is zero (recovering strict anchor velocity), or it can drift the anchor velocity by a small amount governed by `(π/σ_t_src)·∑_m m·a_m`. For M=0 the match is automatic and strict.

This is an additional finding that R1 did not explicitly compute. **F2 does NOT give strict anchor velocity for general M without a constraint on `a_m`.** The constraint `∑_m m·a_m = 0` is one linear equation; to enforce it strictly one could either (i) reparametrise the head to output `a_1, …, a_{M-1}` and set `a_M = −(1/M) ∑_{m<M} m·a_m`, or (ii) accept the soft anchor velocity (analogous to the soft anchor at `τ=1` in §5.3) and let the loss shape it.

#### (b) Along the segment, M = 0, on-trajectory: `x_t = Ψ(σ) = x̂_0 + (σ/σ_t_src)(x_t_src − x̂_0)`

- **F1 along segment**: `(x_t − Ψ(σ))/σ = (Ψ(σ) − Ψ(σ))/σ = 0` for every σ. **Constant zero.** Not `u_src`. So F1 fails to give the right velocity even on the deterministic trajectory it was meant to follow.
- **F2 along segment**: `∂_σ Ψ(σ) = (x_t_src − x̂_0)/σ_t_src = u_src` constant. ✓

The reason for F1’s failure is that F1 only makes sense when `Ψ` is interpreted as an `x_0`-*estimate* (constant or slowly-varying clean-image proxy), not as the trajectory itself. With `Ψ` = trajectory, on-trajectory `x_t = Ψ` and F1 collapses to zero.

DXPolicy avoids this trap by parametrising `denoising_output_x_0` as N candidate `x_0` values that the network outputs at N grid points along the segment ([`piflow_policies/dx.py:51-54`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L51-L54), [`piflow_policies/dx.py:77-101`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L77-L101)). At inference, `x_0(σ)` is interpolated *between those candidates*, and `u = (x_t − x_0(σ))/σ` ([`piflow_policies/dx.py:100`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L100)). DX’s `x_0(σ)` is a different object from Idea 1’s `Ψ(σ)`.

#### (c) SDE-deviated `x_t` (off-trajectory)

The sampler in [`pipeline_gmdit.py:122-138`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L122-L138) is a multi-substep loop within a segment. If the sampler is stochastic (e.g. `FlowSDEScheduler`), it can inject noise that pushes `x_t` off the deterministic trajectory `Ψ(σ)` between substeps.

- **F1 behaviour**: `u_F1 = (x_t − Ψ(σ))/σ` depends linearly on the deviation `x_t − Ψ(σ)`. If `x_t` is above `Ψ(σ)`, `u_F1` is larger (in `ε − x_0` convention, larger `u` → larger `ε` content → `x_t` heading toward more noise / less data). The integration of this `u_F1` over a small Δσ produces a "catch-up" toward `Ψ(σ)` of magnitude `(x_t − Ψ)/σ · Δσ`. This is the DXPolicy/GMFlowPolicy behaviour ([`piflow_policies/dx.py:100`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L100); [`piflow_policies/gmflow.py:78-93`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/gmflow.py#L78-L93) where the Bayesian posterior also depends on `x_t`).

- **F2 behaviour**: `u_F2 = +∂_σ Ψ(σ)` ignores `x_t` entirely. After SDE noise injection, the next call to the policy gets a different `x_t` but returns the same `u`. Within a segment, the policy gives a *deterministic* drift; the sampler’s stochastic step is solely responsible for the noise.

#### (d) Pick

**Pick F2** for R1.

Justification against (a), (b), (c) jointly:
- (a): F2 matches `u_src` at the anchor strictly for M=0 and softly for M≥1 (the soft case requires `∑_m m·a_m = 0`, which is one linear constraint in the `a_m` head — same status as the `∑_{m odd} b_m` cosine-residual question in §5.3, also handled softly). F1 returns 0 at the anchor, never `u_src`.
- (b): F2 gives constant `u_src` along the M=0 segment; F1 gives 0 everywhere on the deterministic trajectory. F2 is the formula that respects the paper’s "differentiate the function" semantics ([pi_flow_three_ideas.md:44-46](../three-ideas.md#L44-L46)).
- (c): F2 ignores SDE deviation. This is acceptable for R1 because (i) PiFlow’s segment structure already calls the network at every segment boundary, bounding the within-segment SDE-deviation by the segment size; (ii) the F1 behaviour is appropriate for `Ψ = x_0`-estimate (DX/GMFlow), not for `Ψ = trajectory` (Idea 1) — applying F1 to a trajectory `Ψ` produces structurally wrong velocities (b above).

**Hybrid option, deferred**: if SDE-deviation robustness is empirically needed, one can use
$$u_{\text{hybrid}}(x_t, \sigma_t) = \underbrace{\partial_\sigma \Psi(\sigma_t)}_{\text{F2: planned drift}} + \kappa\,\underbrace{\frac{x_t - \Psi(\sigma_t)}{\sigma_t}}_{\text{F1-like correction}}$$
with `κ ∈ [0, 1]` a small coupling constant. On-trajectory the second term vanishes; off-trajectory it acts as a soft pull toward `Ψ(σ)`. **This is not adopted in R1**; defer to a follow-up note if/when empirical testing shows the pure-F2 policy drifts under SDE sampling.

### 5.5 Schedule swap (linear ↔ trig) — preserved with notational fix

Under linear (`α = 1 − σ`) or trig (`α = cos(πt/2)`, `σ = sin(πt/2)`), the trajectory `Ψ(σ)` is parametrised in σ-space directly. The schedule choice enters only through (i) what `σ_t_src` value the segment starts at, and (ii) how the sampler discretises `σ` between segments. The Fourier coefficients `{a_m, b_m}` and the predicted `x̂_0` are schedule-agnostic; they describe the trajectory shape, not the time-discretisation.

The implication: training-time and inference-time schedules need not coincide *if* they place segment boundaries at the same `σ`-values. If they don’t (e.g. trained on linear σ-grid, deployed on trig σ-grid), `Ψ` is still well-defined on each segment but the segments themselves cover different `σ`-ranges. This is fine for elastic inference. (See [ideas-comparison.md:33-49](../ideas-comparison.md#L33-L49) for the parallel issue with the GMM posterior.)

### 5.6 No-collapse failure mode — preserved from R1

If the network ignores the Fourier coefficients and only predicts `x̂_0`, the policy reduces to constant-velocity flow (the M=0 case). This is the expected baseline. If the network *also* sets `x̂_0` poorly, the Fourier coefficients can in principle compensate (they have spatial freedom), but the recommended training loss should regularise `‖a_m‖, ‖b_m‖` so this rerouting does not happen — analogous to DX’s `polynomial` mode regularisation ([`piflow_policies/dx.py:92-97`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L92-L97)).

---

## 6. Notes for the Reviewer

### 6.1 Things I had to assume (revised)

| Assumption | Why I made it | What to push back on |
|---|---|---|
| The formula’s `t` is segment-local `τ ∈ [0,1]`, not global `t/T_max`. | Without this, `X_start, X_end` cannot both be physically meaningful per-segment quantities. | If the paper means global `t`, Option B is forced and the analysis changes substantially. |
| `X_start = x_t_src`, `X_end = x̂_0` (i.e. `(start, end)` = (noise side, data side), aligned with forward generation reading order). | Convention; the paper does not say. The opposite assignment with `τ` reversed is mathematically equivalent. | If the paper means `(start, end)` = (clean, noisy), the formula is the same up to relabelling. |
| `X_end` is a soft anchor (cosine-residual perturbation absorbed by the `x̂_0` prediction or by the loss). | §5.3 derived that strict anchor reproduction at `τ=1` requires `∑_{m odd} b_m = 0`, which is one linear constraint. A soft anchor is the simplest treatment. | If empirical training shows instability, switch to (i) zero-out odd `b_m`, (ii) sin-only basis, (iii) modify `φ_1`. |
| Anchor velocity at `τ=0` is soft for M ≥ 1 (requires `∑_m m·a_m = 0` for strict match). | §5.4(a) derived this. The constraint is one linear equation. | If the GMFlow strict anchor-velocity property is required, parametrise `a_M` as `−(1/M)∑_{m<M} m·a_m`. |
| The Fourier policy parametrises the *trajectory* `Ψ(σ) ≈ x_t(σ)`, with velocity recovered by F2 (`u = +∂_σ Ψ`). | This is the only interpretation consistent with [pi_flow_three_ideas.md:44-46](../three-ideas.md#L44-L46) ("differentiate the function") and the codebase’s `u = +∂_σ x_t` convention ([`gaussian_flow.py:90-95`](../../repos/piFlow/lakonlab/models/diffusions/gaussian_flow.py#L90-L95), [`gaussian_flow.py:138`](../../repos/piFlow/lakonlab/models/diffusions/gaussian_flow.py#L138)). The alternative interpretation (`Ψ` = `x_0`-estimate, F1) reproduces DX/GMFlow but requires re-reading the paper. | Confirm with the source paper. |
| `u_to_x_0` ([`gmflow.py:153-183`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L153-L183)) is silently linear-schedule. | Reading the code: `x_0 = x_t − σ·u` is exact only when `α = 1 − σ`. **Out of scope for R1; preserved here for a separate R-task.** | (Investigate separately; do not fix in R1.) |
| The user’s rollout plan (`derivations/03_rollout_plan.md`) is orthogonal to Idea 1. | Idea 1 *replaces* the GMM posterior with a parametric trajectory, so it does not consume `gmflow_posterior_mean_jit_general`. | If GMFlow becomes the VP baseline for comparison, the wrapper needs to be production-routed (Phase 4 of the rollout, blocked on Gate (g) at [derivations/03_rollout_plan.md:75-86](../../derivations/03_rollout_plan.md#L75-L86)). |

### 6.2 Open questions (revised — items resolved by R1.1 marked done)

1. ~~**Position vs velocity parametrisation.**~~ **Resolved (B2, §5.4):** `Ψ` is the trajectory; velocity is `u = +∂_σ Ψ` (F2).
2. ~~**τ-axis direction.**~~ **Resolved (B1, §4, §5.1):** `τ(σ) = 1 − σ/σ_t_src`. Derived from the M=0 collapse.
3. ~~**Boundary-term construction.**~~ **Resolved (§5.3):** accept soft `X_end` anchor at `τ=1`. Strict-anchor remedies deferred to a follow-up.
4. **Loss for `x̂_0` and `{a_m, b_m}`.** Single trajectory-fitting loss vs separate `x̂_0` supervision + coefficient regularisation. Single-loss is cleaner but may underweight `x̂_0`. **Open.**
5. **SDE-deviation robustness.** Pure F2 ignores `x_t` deviation; the F2-plus-correction hybrid in §5.4(d) is available. **Open** — pick after empirical testing.
6. **Anchor-velocity strict-match (§5.4(a))**: strict match at the anchor for M ≥ 1 requires `∑_m m·a_m = 0`. Decide whether to enforce it via reparametrisation or accept soft. **Open.**

### 6.3 What I did *not* check

- Whether the source paper’s Sec. 7 actually specifies the boundary terms `φ_0, φ_1` or fixes the τ-direction. I worked from the formulas in [pi_flow_three_ideas.md:37](../three-ideas.md#L37) and [ideas-comparison.md:25](../ideas-comparison.md#L25); the paper itself is not in the context I was given.
- Numerical stability of F2 under VP at small `σ`. F2 does not divide by σ, so the obvious `1/σ` blowup that worried R1 is absent. But `∂_τ Ψ` is bounded by something like `‖x_t_src − x̂_0‖ + π·∑_m m·(‖a_m‖ + ‖b_m‖)`, which scales linearly with `M`; high-M can produce noisy velocities even without `1/σ`. Quantification is out of scope here.
- Whether the network architecture (`GMDiTTransformer2DModel` at [`pipeline_gmdit.py:11`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L11)) has a convenient way to add `2M + 1` extra output heads. A separate audit of `GMDiTTransformer2DModel` is needed before sizing `M`.
- Distillation training procedure. Idea 1 might be trained from scratch or distilled from a GMFlow teacher; the choice affects what `x̂_0` is supervised against. Out of scope here.

### 6.4 Confidence (revised)

- **High** confidence that Option A is the right structural choice.
- **High** confidence on `τ(σ) = 1 − σ/σ_t_src` (§4 step-by-step derivation; M=0 collapse forces it uniquely with the chosen `(start, end)` reading).
- **High** confidence that F2 is the right velocity-recovery formula given Idea 1’s "differentiate the function" semantics. F1 is the right formula for DX/GMFlow’s `Ψ = x_0`-estimate parametrisation, which is a different policy family — not Idea 1.
- **Medium** confidence on whether the soft anchors (`X_end` cosine-residual and `M ≥ 1` anchor-velocity) are acceptable in practice. Both can be made strict at the cost of one linear constraint per case; deferred.
- **Medium** confidence on SDE-deviation behaviour. Pure F2 is the cleaner R1 choice; the F2-plus-correction hybrid (§5.4(d)) is the obvious next step if needed.

---

## Changes from prior version (R1 → R1.1)

### Bugs fixed

- **B1 (τ-axis direction).** R1 §4 stated `τ(σ) = 1 − σ/σ_t_src`; R1 §5.1 then asserted "the corrected τ = σ/σ_t_src" and propagated the reversal into §5.3, §5.4, §6. R1.1 resolves this with a step-by-step M=0 collapse derivation (§4 and §5.1) and concludes that R1 §4 was correct: **`τ(σ) = 1 − σ/σ_t_src`**. R1’s §5.1 "correction" was a relabelling, not a different convention — the trajectory equation is the same; the apparent disagreement was notational. R1.1 uses the `(1 − τ, τ)` linear basis with `X_start = x_t_src`, `X_end = x̂_0` consistently throughout.

- **B2 (velocity-recovery formula).** R1 §4 used F1 (`u = (x_t − Ψ)/σ`); R1 §5.4 wavered between F1 and F2 and used the wrong sign on F2 (`u = −∂_σ Ψ`). R1.1 derives the codebase velocity convention from [`gaussian_flow.py:90-95`](../../repos/piFlow/lakonlab/models/diffusions/gaussian_flow.py#L90-L95) and [`gaussian_flow.py:138`](../../repos/piFlow/lakonlab/models/diffusions/gaussian_flow.py#L138): along the deterministic trajectory, `u = +∂_σ x_t`. So **F2 is `+∂_σ Ψ`**, not `−∂_σ Ψ`. R1.1 evaluates F1 and F2 at the anchor (a), along the M=0 segment (b), and under SDE deviation (c), and picks **F2**. F1 fails (a) and (b) when `Ψ` is the trajectory, because on-trajectory `x_t = Ψ` makes F1 ≡ 0. F1 is the right formula for DX/GMFlow’s `Ψ = x_0`-estimate parametrisation, which is structurally different from Idea 1.

### Smaller items resolved

- **Cosine-residual at `τ=1` (§5.3).** Picked option (3) from R1’s aside: accept `X_end` as a soft anchor. Rationale in §5.3.
- **`u_to_x_0` linear-schedule finding (§6.1).** Confirmed preserved as a separate R-task; **not** investigated or fixed in R1.1.

### New finding (not in R1)

- **Anchor velocity at `τ=0` for M ≥ 1 is soft (§5.4(a)).** Strict match to `u_src` at the anchor requires `∑_m m·a_m = 0`. R1 did not compute this. R1.1 flags it as item 6 in §6.2 and offers a reparametrisation remedy if strict-match is needed.

### Sections rewritten

- §4 (recommendation) — full rewrite, with derivations for B1 and B2.
- §5.1 (M=0 collapse) — full rewrite as the load-bearing B1 derivation.
- §5.3 (anchor reproduction) — full rewrite under the corrected τ-direction; resolved soft-anchor decision.
- §5.4 (velocity recovery) — full rewrite as the load-bearing B2 derivation.

### Sections lightly edited

- §1.2 (added the `u = +∂_σ x_t` derivation, citing forward process and loss target).
- §1.4 (added the observation that DX and GMFlow both parametrise `x_0`-estimates via F1, distinguishing them from Idea 1’s trajectory parametrisation).
- §1.6 (narrowed the τ-direction assumption to point at §4).
- §2 Option A (velocity bullet now states F2 explicitly; segment-embedding bullet corrected to use the resolved τ-direction).
- §2 boundary-term aside (notes the resolution).
- §3 table (Inference and Robustness rows updated to reflect F2; soft-anchor resolution noted in segment-distillation row).
- §6.1 (assumption table revised: τ-direction and velocity-recovery removed as open assumptions; soft-anchor and anchor-velocity-soft-match added).
- §6.2 (open questions 1–3 marked resolved with section pointers; 4–6 remain open).
- §6.4 (confidences updated).

### Citations preserved (all re-verified)

[`gmflow.py:51-52`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L51-L52), [`gmflow.py:79-139`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L79-L139), [`gmflow.py:147`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L147), [`gmflow.py:161`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L161), [`gmflow.py:185-252`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L185-L252), [`gmflow.py:206`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L206), [`gmflow.py:212`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L212), [`gmflow.py:439-452`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L439-L452), [`gmflow.py:454-474`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L454-L474); [`gaussian_flow.py:90-95`](../../repos/piFlow/lakonlab/models/diffusions/gaussian_flow.py#L90-L95), [`gaussian_flow.py:138`](../../repos/piFlow/lakonlab/models/diffusions/gaussian_flow.py#L138); [`pipeline_gmdit.py:11`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L11), [`pipeline_gmdit.py:60-64`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L60-L64), [`pipeline_gmdit.py:80-138`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L80-L138), [`pipeline_gmdit.py:81`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L81), [`pipeline_gmdit.py:87-91`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L87-L91), [`pipeline_gmdit.py:122-138`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L122-L138), [`pipeline_gmdit.py:123`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L123), [`pipeline_gmdit.py:135`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L135); [`piflow_policies/base.py:7`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/base.py#L7); [`piflow_policies/gmflow.py:8-13`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/gmflow.py#L8-L13), [`piflow_policies/gmflow.py:33-47`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/gmflow.py#L33-L47), [`piflow_policies/gmflow.py:40-47`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/gmflow.py#L40-L47), [`piflow_policies/gmflow.py:60-94`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/gmflow.py#L60-L94), [`piflow_policies/gmflow.py:74-75`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/gmflow.py#L74-L75), [`piflow_policies/gmflow.py:78-92`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/gmflow.py#L78-L92), [`piflow_policies/gmflow.py:78-93`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/gmflow.py#L78-L93), [`piflow_policies/gmflow.py:93`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/gmflow.py#L93); [`piflow_policies/dx.py:13-19`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L13-L19), [`piflow_policies/dx.py:23-46`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L23-L46), [`piflow_policies/dx.py:51-54`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L51-L54), [`piflow_policies/dx.py:51-101`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L51-L101), [`piflow_policies/dx.py:77-101`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L77-L101), [`piflow_policies/dx.py:92-97`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L92-L97), [`piflow_policies/dx.py:100`](../../repos/piFlow/lakonlab/models/diffusions/piflow_policies/dx.py#L100); [pi_flow_three_ideas.md:37](../three-ideas.md#L37), [pi_flow_three_ideas.md:44-46](../three-ideas.md#L44-L46); [ideas-comparison.md:25](../ideas-comparison.md#L25), [ideas-comparison.md:33-49](../ideas-comparison.md#L33-L49), [ideas-comparison.md:49](../ideas-comparison.md#L49); [derivations/03_rollout_plan.md:14-23](../../derivations/03_rollout_plan.md#L14-L23), [derivations/03_rollout_plan.md:14-39](../../derivations/03_rollout_plan.md#L14-L39), [derivations/03_rollout_plan.md:65-72](../../derivations/03_rollout_plan.md#L65-L72), [derivations/03_rollout_plan.md:75-86](../../derivations/03_rollout_plan.md#L75-L86).

### Citations removed

- R1 §6.3 cited [`gmflow.py:113-114`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L113-L114) for `clamp(min=eps)` as the small-σ stability mechanism. Under F2 the `1/σ` divergence is absent, so this citation no longer carries an argument; removed.

---

## C1 — schedule-agnosticism revisited (2026-05-02)

R1.1 §4.3 #5 and §5.5 contain a schedule-agnosticism argument that, on close reading, conflates two distinct claims. This addendum separates them, identifies which one survives, and proposes minimal edits. The basis-choice analysis in [R4 §4](idea1-basis-choice.md#4-quantify-the-residual-under-each-basis) already quantified the failure for one specific case; this section names what was being computed.

### (a) What R1.1 actually asserts (and what it does not)

The two sentences under test are:

> §4.3 #5: "Under any α(σ) schedule, the trajectory is parametrised in σ via τ(σ); the schedule choice enters only through how the sampler maps physical t to σ. Under linear (`α = 1 − σ`) or trig (`α² + σ² = 1`), the same `{x̂_0, a_m, b_m}` define the same `Ψ(σ)` — what changes is the time-discretisation the sampler picks."

> §5.5: "The Fourier coefficients `{a_m, b_m}` and the predicted `x̂_0` are schedule-agnostic; they describe the trajectory shape, not the time-discretisation. The implication: training-time and inference-time schedules need not coincide *if* they place segment boundaries at the same `σ`-values."

These are two separable claims:

- **Claim A (formula-level).** *Given* numerical coefficient values $\{\hat x_0, a_m, b_m\}$ and a segment anchor $(x_{t_{src}}, \sigma_{t_{src}})$, the formula $\Psi(\sigma) = (1-\tau(\sigma))\,x_{t_{src}} + \tau(\sigma)\,\hat x_0 + \sum_m (\ldots)$ is the same function of σ regardless of which schedule produced $x_{t_{src}}$ in the first place. Trajectory math is schedule-blind.
- **Claim B (model-level).** A network $G_\theta$ trained under one schedule produces, on cross-schedule inputs, coefficients whose induced $\Psi(\sigma)$ correctly approximates the deployment schedule's underlying trajectory.

R1.1's wording reads like Claim A (which is true and trivial: it is a property of substituting numbers into a formula) but is *used* to support Claim B (which is the load-bearing claim — the only one that justifies the operational sentence "training-time and inference-time schedules need not coincide"). R1.1 does not separate them.

### (b) Under S2 (train linear, deploy VP): what holds, what breaks

Two sources of cross-schedule mismatch, both load-bearing:

**(i) Conditional distribution of $x_{t_{src}}$ at fixed $\sigma_{t_{src}}$ differs.** The forward process at training is hardcoded `x_t = (1-σ)·x_0 + σ·ε` ([`gaussian_flow.py:90-95`](../../repos/piFlow/lakonlab/models/diffusions/gaussian_flow.py#L90-L95)), with conditional mean $E[x_{t_{src}} \mid x_0, \sigma_{t_{src}}] = (1-\sigma_{t_{src}})\,x_0$. Under VP deployment, the conditional mean is $\sqrt{1-\sigma_{t_{src}}^2}\,x_0$. These differ for every $\sigma_{t_{src}} \in (0, 1)$. So at deployment under VP, the network is queried on inputs drawn from $p_{\text{VP}}(x_{t_{src}} \mid \sigma_{t_{src}})$ that lie outside the training distribution $p_{\text{lin}}(\cdot \mid \sigma_{t_{src}})$.

The network's output on out-of-distribution inputs is, in general, undefined w.r.t. the training signal. The network may produce coefficients that look reasonable, or not.

**(ii) Even if the network had been trained schedule-aware, the optimal coefficients differ across schedules.** R4 §4.1 derived that under perfect prediction $\hat x_0 = x_0$, the M=0 trajectory $\Psi_{B0}^{M=0}(\sigma)$ deviates from the natural VP trajectory by

$$\Delta_{B0,\text{VP}}(\sigma) = x_0 \cdot \bigl[(1 - \sqrt{1-\sigma^2}) - (\sigma/\sigma_{t_{src}})(1 - \sqrt{1-\sigma_{t_{src}}^2})\bigr]$$

with magnitude up to ~17% of $\|x_0\|$ at $\sigma_{t_{src}} \approx 0.9$ ([R4 §4.1 numerical table](idea1-basis-choice.md)). The network would have to produce non-zero $\{a_m, b_m\}$ to absorb this geometric gap under VP, and **zero** $\{a_m, b_m\}$ under linear (where $\Delta_{B0,\text{lin}} = 0$ at perfect prediction). The optimal coefficient values are therefore *schedule-conditioned*, even setting OOD-input issues aside.

**Synthesis:**

| Component of R1.1's claim | Holds under S2? |
|---|---|
| "the same `{x̂_0, a_m, b_m}` define the same `Ψ(σ)`" — read as Claim A | ✓ holds (formula identity) |
| "the same `{x̂_0, a_m, b_m}` define the same `Ψ(σ)`" — read as "the network outputs the same coefficients regardless of schedule" | ✗ false (the network's output depends on its input, and the input distribution depends on the schedule) |
| "schedule choice enters only through how the sampler maps physical t to σ" | ✗ false at the model level. The sampler-discretisation independence is real, but it is not the only schedule dependency: the *forward process α* enters through training data, which the sentence omits |
| "Fourier coefficients `{a_m, b_m}` and the predicted `x̂_0` are schedule-agnostic" | ✗ false. The *formulas that consume them* are schedule-agnostic (Claim A); the *values the network produces* are not |
| "training-time and inference-time schedules need not coincide *if* they place segment boundaries at the same σ-values" | ✗ false in general. Even at identical σ-grid placement, the network's outputs are conditioned on the training schedule's forward process; matching σ-grids does not match the conditional distribution of $x_{t_{src}}$ |
| "Ψ is still well-defined on each segment" — read as Claim A | ✓ holds (well-definedness ≠ correctness) |
| Sampler-discretisation independence within a fixed schedule (R1.1 §4.3 #6, "elastic inference") | ✓ holds — this is the *real* schedule-agnosticism, scoped correctly |

The substantive claim that "training-time and inference-time schedules need not coincide" does not survive. What survives is the weaker but still useful claim that **within a fixed schedule, the policy is robust to changes in the σ-grid the sampler picks**.

### (c) Proposed minimal edits to R1.1

Decision: **do not retract §4.3 #5 or §5.5 entirely.** A meaningful schedule-agnosticism does survive, but at a smaller scope than R1.1 currently advertises. Edit to scope it correctly.

Proposed edits (listed; not applied here):

**§4.3 #5 — replace second sentence onward.**
- Current: "Under linear (`α = 1 − σ`) or trig (`α² + σ² = 1`), the same `{x̂_0, a_m, b_m}` define the same `Ψ(σ)` — what changes is the time-discretisation the sampler picks. The schedule-agnostic posterior infrastructure ([derivations/03_rollout_plan.md:14-23]) is not consumed here, but coexists cleanly."
- Proposed: "Under any α(σ) schedule, the formula $\Psi(\sigma; \{\hat x_0, a_m, b_m\})$ is the same function of σ for fixed coefficient values. **What is schedule-dependent is the network producing those coefficients**: training under schedule $\mathcal S$ exposes the network to $x_{t_{src}} \sim p_{\mathcal S}(\cdot \mid \sigma_{t_{src}})$, and the optimal coefficients at $(x_{t_{src}}, \sigma_{t_{src}})$ depend on $\mathcal S$ via the natural-trajectory geometry (see [R4 §4](idea1-basis-choice.md)). The Fourier formula's *consumption* of coefficients is schedule-blind; the *production* of coefficients is not. The schedule-agnostic posterior infrastructure ([derivations/03_rollout_plan.md:14-23]) is not consumed here."

**§5.5 — rewrite the "implication" paragraph.**
- Current: "The implication: training-time and inference-time schedules need not coincide *if* they place segment boundaries at the same `σ`-values. If they don't (e.g. trained on linear σ-grid, deployed on trig σ-grid), `Ψ` is still well-defined on each segment but the segments themselves cover different `σ`-ranges. This is fine for elastic inference."
- Proposed: "The implication is narrower than it might appear. **Within a fixed schedule**, the σ-grid the sampler picks may differ between training and inference: the network's coefficients describe a continuous Ψ(σ) that can be evaluated at any σ in the segment, so the policy is robust to substep-count changes (this is the elastic-inference property — analogous to `DXPolicy.segment_size, shift` invariants — [`piflow_policies/dx.py:13-19`]). **Across different schedules** (e.g. trained linear, deployed VP), the policy is **not** robust without retraining: the conditional distribution $p(x_{t_{src}} \mid \sigma_{t_{src}})$ differs across schedules ([`gaussian_flow.py:90-95`] vs the VP forward process), and the optimal coefficients at $(x_{t_{src}}, \sigma_{t_{src}})$ differ correspondingly. R4 §4.1 quantifies the geometric component of this gap (up to ~17% of $\|x_0\|$ for $\sigma_{t_{src}} = 0.9$ at perfect prediction)."

**§6.1 (assumptions table) — add or strengthen one row.**
- Add: "Schedule-agnosticism is at the formula level, not at the model level. Training under one schedule and deploying under another requires retraining; only σ-grid changes within a single schedule are free."

**§6.4 (confidence) — add or amend.**
- Current section claims "Schedule-agnosticism orthogonal" with high confidence. Replace with: "**High** confidence on the formula-level schedule-blindness (Claim A). **Low** confidence on cross-schedule transfer of trained coefficients (Claim B); R4 §4 shows up to 17% systematic geometric residual under linear→VP transfer at perfect prediction, before any OOD-input effects."

### Survives or retract?

Survives, in narrower scope. The Fourier policy *is* schedule-agnostic at the formula level (a property the GMM-posterior route does not have, and which is therefore worth stating). The Fourier policy *is* robust to σ-grid changes within a fixed schedule (elastic inference). The Fourier policy is *not* robust to cross-schedule deployment without retraining; that claim should be retracted from §5.5 and the section should be retitled or rescoped to "σ-grid agnosticism" rather than "schedule agnosticism".

The R4 §4.1 residual table is the empirical face of this section — it quantifies what R1.1's section was implicitly claiming would not happen.

---

## C2/Q6 — anchor velocity at M ≥ 1, strict vs soft (2026-05-02)

R1.1 §5.4(a) derived an anchor-velocity drift $\Delta u = u_{F2}(\sigma_{t_{src}}) - u_{src}$ for $M \ge 1$ under linear basis F2; §6.2 item 6 left the strict-vs-soft choice open. This addendum diagnoses the magnitude under realistic conditions and picks an option.

### (a) Re-derivation of Δu

Compact restatement. With τ = 1 − σ/σ_t_src and the linear-basis residual $\sum_m a_m\sin(m\pi\tau) + b_m(\cos(m\pi\tau)-1)$:

$$\partial_\tau \Psi(0) \;=\; -x_{t_{src}} + \hat{x}_0 \;+\; \pi \sum_{m=1}^{M} m\,a_m \quad\text{(sin' contributes }m\pi\cos(0)=m\pi;\;\cos'\text{ vanishes at 0)}.$$

F2 anchor velocity:
$$u_{F2}(\sigma_{t_{src}}) \;=\; +\partial_\sigma \Psi(\sigma_{t_{src}}) \;=\; -\tfrac{1}{\sigma_{t_{src}}}\,\partial_\tau \Psi(0) \;=\; \underbrace{\tfrac{x_{t_{src}} - \hat{x}_0}{\sigma_{t_{src}}}}_{u_{src}} \;-\; \underbrace{\tfrac{\pi}{\sigma_{t_{src}}}\sum_m m\,a_m}_{-\Delta u}.$$

So $\boxed{\Delta u \;=\; -\dfrac{\pi}{\sigma_{t_{src}}}\sum_{m=1}^{M} m\,a_m}$. R1.1 §5.4(a) is confirmed (verification V4 of the M=2 case is in [`idea1-basis-choice.md`](idea1-basis-choice.md) verification log). The drift carries a $1/\sigma_{t_{src}}$ factor — distinct from the $1/\sigma$-at-evaluation factor that R1.1 §4.3 #7 was discussing.

### (b) Magnitude bounds on Δu

**Random init.** Treat $a_m$ as i.i.d. with std $\sigma_a$. Then $\text{Var}(\sum m\,a_m) = \sigma_a^2 \sum_{m=1}^{M} m^2 = \sigma_a^2 \cdot M(M+1)(2M+1)/6 \sim \sigma_a^2\,M^3/3$, so

$$\bigl|\textstyle\sum_m m\,a_m\bigr|\big|_{\text{init}} \;\sim\; \frac{\sigma_a\,M^{3/2}}{\sqrt 3} \;,\qquad |\Delta u|_{\text{init}} \;\sim\; \frac{\pi\,\sigma_a\,M^{3/2}}{\sqrt 3\,\sigma_{t_{src}}}.$$

Concrete: $M=4,\; \sigma_a = 1,\; \sigma_{t_{src}} = 0.01 \Rightarrow |\Delta u| \approx 1450$. For comparison, near the data endpoint $\|x_{t_{src}} - \hat x_0\| \to 0$ so $|u_{src}|$ is small (often $\ll 100$). **Random-init drift dominates $u_{src}$ at deep segments by orders of magnitude.** This is expected — random init is not a deployment regime, but it sets a worst-case for what the loss must control.

**Trained, well-conditioned.** Spectral coefficients of smooth functions decay; suppose $|a_m| \sim m^{-p}$. Then $|m\,a_m| \sim m^{-(p-1)}$ and

$$\textstyle\sum_{m=1}^M m\,a_m \;\sim\; \begin{cases} M^{2-p}/(2-p) & p < 2 \\ \ln M & p = 2 \\ \zeta(p-1) & p > 2 \text{ (bounded as }M\to\infty\text{)} \end{cases}$$

For exponential decay $|a_m| \sim e^{-cm}$, $\sum m\,e^{-cm}$ converges to $e^{-c}/(1-e^{-c})^2$; for $c = 1$ this is $\approx 0.92$.

**Threshold:** $\sum m\,a_m$ stays bounded as M grows iff **p > 2** (or faster decay). Spectral approximations of analytic functions achieve exponential decay; of $C^k$ functions achieve $p \approx k+1$. So a smooth target with $C^2$ residual already gives bounded $\sum m\,a_m$.

But "bounded as M grows" is not the relevant condition. The relevant question is: **at fixed M (4–8 in practice), how big is $|\sum m\,a_m|$ after training, and is $(\pi/\sigma_{t_{src}})\cdot|\sum|$ comparable to $|u_{src}|$ at deep segments?**

Order-of-magnitude. For trained coefficients with $|a_m| \sim 1/m^2$ and $M = 8$: $\sum m\,a_m = \sum 1/m \approx H_8 \approx 2.7$. At $\sigma_{t_{src}} = 0.01$: $|\Delta u| \approx \pi \cdot 2.7/0.01 \approx 850$. With faster decay ($p = 3$): $\sum 1/m^2 \approx 1.55$, $|\Delta u| \approx 490$. For $|u_{src}|$ to be comparable, $\|x_{t_{src}} - \hat x_0\|$ would need to be $\sim 5$ or more — implausibly large at $\sigma_{t_{src}} = 0.01$.

**Conclusion on Q-magnitude.** Even under favourable trained-coefficient assumptions, $|\Delta u|$ at the anchor of deep segments ($\sigma_{t_{src}} \lesssim 0.05$) is **comparable to or larger than $|u_{src}|$**. R1.1 §4.3 #7's "no 1/σ blowup" claim is correctly scoped (it was about evaluation-σ, not anchor-σ), but the operational implication that F2 is universally well-behaved at small σ is **wrong**: F2 has its own σ-blowup, just at the segment anchor instead of the substep evaluation point. The two blowups affect different axes but neither is universally absent.

### (c) Reparametrisation cost (Option R)

Option R sets $a_M = -(1/M)\sum_{m<M} m\,a_m$ algebraically, enforcing $\sum_m m\,a_m = 0$ by construction.

| Aspect | Cost |
|---|---|
| Head output dim | 2M → 2M − 1. One $C\cdot H \cdot W$ map dropped. Negligible memory savings; no computational impact. |
| Per-query op | One weighted sum: $a_M = -\tfrac{1}{M}(a_1 + 2 a_2 + \ldots + (M{-}1)a_{M-1})$. Pointwise tensor arithmetic. |
| `torch.jit` compatibility | Identical pattern to inline arithmetic in [`gmflow_posterior_mean_jit_general` (gmflow.py:79-139)](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L79-L139), which already handles linear combinations of tensor outputs in JIT mode. No issue. |
| Interaction with τ=1 soft anchor (R1.1 §5.3) | Independent: τ=1 strict would require $\sum_{m\text{ odd}} b_m = 0$ — linear in $b$, orthogonal to the linear-in-$a$ drift constraint. Both can be enforced simultaneously by reparametrising one $a_m$ and one $b_m$. R4 picked B0 with soft τ=1, so the b-constraint is currently inactive; if reactivated it does not interfere with R. |
| Degrees of freedom | $M$ degrees → $M-1$ for $a$. At M=4 the loss is 25%; at M=8, 12.5%. Modest. |

R has effectively no implementation cost. The trade-off is solely about whether the constraint is empirically too restrictive.

### (d) Pick: S (with a clear flip-trigger)

**Pick Option S (accept soft drift) for shipping. Flip to R if the empirical trigger fires.**

Justification:

1. **The drift is finite, not divergent.** $\Delta u$ scales as $1/\sigma_{t_{src}}$, but so does $u_{src}$. Both are well-defined for any $\sigma_{t_{src}} > 0$ used by the production sampler ([`gaussian_flow.py:191-195`](../../repos/piFlow/lakonlab/models/diffusions/gaussian_flow.py#L191-L195) flow_shift heuristic plus the [`gmflow.py:113-114`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L113-L114) `clamp(min=eps)` ensure $\sigma_{t_{src}} \ge \text{eps}$ in practice). The blowup is structural at $\sigma_{t_{src}} = 0$, but that's not a sampled point.

2. **The training loss has the right gradient to suppress drift.** The trajectory loss penalizes deviation from the true x_t along the segment. Large $|\sum m\,a_m|$ produces a sharp velocity at $\tau = 0$ that propagates inward as a curved trajectory; if the truth is a smoother curve (which, under linear schedule, it is — see R4 §4.1), the loss penalises the sharpness. So the drift is *not* free for the network to take; it incurs trajectory-loss penalty in the σ-neighbourhood of the anchor. Empirically we expect $|\sum m\,a_m|$ to be small at convergence under linear schedule.

3. **R has no killer-app upside.** R is cheap (§(c)) but not cost-free in expressiveness. With M small (4–8), constraining one $a_m$ removes a meaningful fraction of the residual capacity. Under linear schedule (production), the soft-anchor M=0 case is *exactly* correct (R4 §4.5: zero geometric residual under linear), and the residual coefficients only need to absorb the prediction-error component of the deviation. A constraint that halves expressiveness at M=2 in exchange for a property the loss already pulls toward is not obviously a win.

4. **Symmetry with the τ=1 soft-anchor decision.** R1.1 §5.3 chose to leave τ=1 soft for the analogous reason ($x̂_0$ is a network prediction, the loss shapes it). The same logic applies to the anchor velocity: $u_{src}$ is implicit in the network's $x̂_0$ prediction; letting the residual coefficients add a small offset is consistent with the soft-everything-except-strictly-required policy.

**Empirical trigger to flip to R:** at training convergence under linear schedule, monitor the histogram of $|\sum_m m\,a_m|$ across the training set (or a held-out validation set), per-pixel. Concretely: define
$$D := \text{median}_{\xi}\,\bigl|\textstyle\sum_m m\,a_m(\xi)\bigr|.$$
Flip to R if **$D > 0.1 \cdot \|x_{t_{src}} - \hat x_0\|$ at the deepest production segment** (i.e. if drift is more than 10% of $u_{src}$ in magnitude after multiplication by $\pi/\sigma_{t_{src}}$). Practically: $|\Delta u|/|u_{src}| > 0.1$ at the deepest sampled $\sigma_{t_{src}}$.

A second flip-trigger: **anchor-discontinuity artefacts in samples.** If reconstructions show ringing or discontinuity at segment boundaries (visible in the σ-neighbourhood of each anchor where the drift is largest), the drift is hurting trajectory quality and R is warranted regardless of the $D$ statistic.

### (e) Confidence

- **High** confidence on the (a) re-derivation (V4 verified the M=2 case symbolically; M ≥ 1 generalises directly).
- **Medium** confidence on (b) bounds. Random-init bound is rigorous (i.i.d. variance argument). Trained-coefficient bound is conditional on a spectral-decay assumption that I have not measured against actual training of this model family. The threshold $p > 2$ for M-independent $\sum m\,a_m$ is a clean theoretical line, but whether trained Idea-1 networks land at $p > 2$ is empirical.
- **Medium-high** confidence on the recommendation S. The decision is reversible (R is one tensor op away from S), so the cost of being wrong is small.
- **Low** confidence that the drift is benign at $\sigma_{t_{src}} \lesssim 0.01$ specifically. Order-of-magnitude estimates suggest the drift can dominate at very deep segments. This is the regime where the flip-triggers would fire; instrument early.

### (f) Implication for R1.1's wording

R1.1 §4.3 #7 should be qualified: "no 1/σ blowup at the substep evaluation point" is the correct claim; the F2 anchor velocity has a separate $1/\sigma_{t_{src}}$ factor whenever the residual $\sum m\,a_m \ne 0$. R1.1's text is technically correct but rhetorically misleading. Minimal edit (listed, not applied here): add a sentence to §4.3 #7 — "**Caveat:** the F2 anchor velocity at $\sigma = \sigma_{t_{src}}$ contains a residual $-(\pi/\sigma_{t_{src}})\sum_m m\,a_m$ that scales as $1/\sigma_{t_{src}}$ when $\sum m\,a_m \ne 0$ (R1.1 §5.4(a), C2 below); the no-blowup claim refers to evaluation-σ, not anchor-σ."

---

## Q5 — SDE-deviation: pure F2 vs hybrid (2026-05-02)

R1.1 §5.4(c)-(d) compared pure F2 against the hybrid F2-plus-correction $u = \partial_\sigma\Psi + \kappa(x_t - \Psi)/\sigma$ for handling SDE-sampler noise injections, picked pure F2, and deferred the hybrid. R1.1 §6.2 item 5 left this as an open question; §6.4 marked it medium confidence. This addendum bounds the SDE deviation, costs the hybrid, and decides.

### (a) Q-magnitude — per-substep and cumulative SDE deviation

The SDE step in the production sampler is in [`flow_sde.py:166`](../../repos/piFlow/lakonlab/models/diffusions/schedulers/flow_sde.py#L166):

```
prev_sample = α_to·x_0 + σ_to·(m·ε + √(1-m²)·η)
```

with $m = (\sigma_{\text{to}}\,\alpha)/(\sigma\,\alpha_{\text{to}})^{h^2}$ ([line 162-164](../../repos/piFlow/lakonlab/models/diffusions/schedulers/flow_sde.py#L162-L164)) and default $h = 1.0$ ([line 30](../../repos/piFlow/lakonlab/models/diffusions/schedulers/flow_sde.py#L30)). Fresh noise $\eta \sim \mathcal N(0, I)$ per substep. (`FlowMapSDEScheduler` defaults to $h=0$ → deterministic; the SDE-deviation question only matters under `FlowSDEScheduler` with $h > 0$.)

**Per-substep noise std.** For small substep ($\sigma_{\text{to}} \approx \sigma$), $m \approx 1$ and $1 - m^2 \approx 2(1-m)$. Linearising under linear schedule ($\alpha = 1-\sigma$):
$$1 - m \;\approx\; \frac{\Delta\sigma_{\text{sub}}}{K}\,\Bigl(\frac{1}{\sigma} + \frac{1}{\alpha}\Bigr),\qquad \text{noise std per substep} \;\approx\; \sigma_{\text{to}}\sqrt{1-m^2} \;\approx\; \sigma\sqrt{\tfrac{2\Delta\sigma_{\text{sub}}}{K}\Bigl(\tfrac{1}{\sigma}+\tfrac{1}{\alpha}\Bigr)}.$$

**Cumulative deviation over K substeps in a segment.** Independent gaussians sum in variance. With per-substep std $\sim \sigma_{\text{avg}}\cdot\sqrt{1-m^2}$ and $\sqrt{1-m^2} \sim \sqrt{2\Delta\sigma_{\text{seg}}/(K\sigma_{\text{avg}})}$ (for $\sigma \approx \alpha$ regime), cumulative std

$$|\Delta|_K \;\sim\; \sqrt{K \cdot \sigma_{\text{avg}}^2 \cdot 2\Delta\sigma_{\text{seg}}/(K\sigma_{\text{avg}})} \;=\; \sqrt{2\,\sigma_{\text{avg}}\,\Delta\sigma_{\text{seg}}}.$$

**The cumulative deviation is independent of K** — the K-th-root averaging from many substeps exactly cancels the per-substep variance scaling. The deviation is set by segment geometry alone.

**Concrete:** segment from $\sigma = 0.5$ to $\sigma = 0.25$ ($\sigma_{\text{avg}} = 0.375$, $\Delta\sigma_{\text{seg}} = 0.25$): $|\Delta| \approx \sqrt{0.19} \approx 0.43$ per pixel std.

For comparison, the within-segment trajectory length is $|\partial_\sigma \Psi \cdot \Delta\sigma_{\text{seg}}| \sim |u_{src}|\cdot\Delta\sigma_{\text{seg}}$. With $\|x_{t_{src}} - \hat x_0\| \sim 1$ (unit-variance latents), $|u_{src}| \sim 1/\sigma_{t_{src}} = 2$, so trajectory length $\sim 0.5$. **SDE deviation ≈ trajectory length.** This is the standard SDE-sampling regime — noise is comparable to drift per step.

### (b) Q-pure-F2-error — trajectory-error accumulation

Under pure F2, $u = \partial_\sigma\Psi(\sigma)$, x_t-independent. The policy emits $x_0^{\text{policy}}(\sigma) = x_t - \sigma\,\partial_\sigma\Psi(\sigma) = (\text{natural }x_0) + \Delta$, where $\Delta = x_t - \Psi(\sigma)$ is the current SDE deviation.

After one SDE step ([`flow_sde.py:166`](../../repos/piFlow/lakonlab/models/diffusions/schedulers/flow_sde.py#L166)):
$$x_t^{(k+1)} = \alpha_{\text{to}}((\text{natural }x_0) + \Delta^{(k)}) + \sigma_{\text{to}}(m\,\varepsilon + \sqrt{1-m^2}\,\eta) = \Psi(\sigma_{\text{to}}) + \alpha_{\text{to}}\,\Delta^{(k)} + \sigma_{\text{to}}\sqrt{1-m^2}\,\eta.$$

So
$$\boxed{\Delta^{(k+1)} = \alpha_{\text{to}}\,\Delta^{(k)} + (\text{new noise}).}$$

For $\sigma_{\text{to}}$ small, $\alpha_{\text{to}} \approx 1$: $\Delta^{(k+1)} \approx \Delta^{(k)} + (\text{new noise})$. **Pure F2 has no catch-up — deviation persists with multiplicative factor near 1, plus fresh noise each substep.** After K substeps:
$$\Delta^{(K)} \approx \prod_{k}\alpha_{\text{to}}^{(k)}\cdot\Delta^{(0)} + \text{(cumulative noise)} \approx \Delta^{(0)} + |\Delta|_K.$$

**Fraction of trajectory length:** $|\Delta^{(K)}|/|u_{src}\cdot\Delta\sigma_{\text{seg}}| \sim |\Delta|_K \cdot \sigma_{t_{src}}/(\|x_{t_{src}} - \hat x_0\|\cdot\Delta\sigma_{\text{seg}})$. For the example above (~0.5 / 0.5): the deviation is ~100% of trajectory length within a segment, **O(1) and independent of K**. The pure-F2 trajectory at substep K has roughly 50% chance of being on the "wrong side" of $\Psi$ relative to the true SDE path.

**Inter-segment recovery.** The next segment's network call uses the *deviated* $x_t^{(K)}$ as the new $x_{t_{src}}$ ([`pipeline_gmdit.py:81-91`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L81-L91)), so the policy re-anchors to the actual sampler state. Errors do not compound across segments. The pure-F2 issue is purely *intra-segment*.

### (c) Q-hybrid-cost — does κ·Δ/σ blow up?

Hybrid: $u_{\text{hybrid}}(x_t, \sigma) = \partial_\sigma\Psi(\sigma) + \kappa(x_t - \Psi(\sigma))/\sigma$. The catch-up term scales as $1/\sigma$ in u-space. **In x_0-space** (which is what the sampler consumes via `prediction_type='x0'` at [`pipeline_gmdit.py:138`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L138)):
$$x_0^{\text{hybrid}} = x_t - \sigma\,u_{\text{hybrid}} = (\text{natural }x_0) + (1-\kappa)\Delta.$$
The σ multiplication cancels the 1/σ in the catch-up term. **Pipeline-safe.**

Substituting into the SDE step:
$$\Delta^{(k+1)} = \alpha_{\text{to}}(1-\kappa)\Delta^{(k)} + (\text{new noise}).$$
For $\kappa = 1$, $\Delta^{(k+1)} = (\text{new noise})$ — full reset per substep. For $\kappa = 0$, recover pure F2.

**1/σ blowup** in the policy's u-output (not pipeline-relevant, but flagged by R1.1 §4.3 #7 as a structural concern): at the smallest sampled σ, $\sigma_{\min} \ge \text{eps} \approx 10^{-6}$ via [`gmflow.py:113-114`](../../repos/piFlow/lakonlab/models/diffusions/gmflow.py#L113-L114) `clamp(min=eps)`. With $|\Delta| \sim 0.4$ from (a), $\kappa = 0.5$: $|\kappa\Delta/\sigma_{\min}| \sim 2\times 10^5$ — astronomically large. But this never reaches a sampler step (the sampler doesn't probe $\sigma = \text{eps}$ directly; the smallest **sampled** σ is the segment endpoint, typically $\gtrsim 10^{-3}$ in production).

At a realistic small evaluation σ = 0.01 with $|\Delta| = 0.04$, $\kappa = 0.5$: $|\kappa\Delta/\sigma| = 2$, vs $|\partial_\sigma\Psi| \sim |u_{src}| \sim 1/\sigma_{t_{src}}$. For deep segments ($\sigma_{t_{src}} \sim 0.05$), $|u_{src}| \sim 20$, so catch-up is ~10% of planned drift — reasonable. For shallow segments, ratio is larger but still bounded.

**Threshold where catch-up = planned drift:** $\kappa|\Delta|/\sigma = |\partial_\sigma\Psi|$. With $|\partial_\sigma\Psi| \sim |u_{src}|\sim \|x_{t_{src}}-\hat x_0\|/\sigma_{t_{src}}$ and $|\Delta| \sim \sqrt{\sigma_{t_{src}}\Delta\sigma_{\text{seg}}}$: dominance happens when $\kappa\sqrt{\sigma_{t_{src}}\Delta\sigma_{\text{seg}}}/\sigma \gtrsim \|x_{t_{src}}-\hat x_0\|/\sigma_{t_{src}}$, i.e. at $\sigma \lesssim \kappa\sigma_{t_{src}}\sqrt{\sigma_{t_{src}}\Delta\sigma_{\text{seg}}}/\|x_{t_{src}}-\hat x_0\|$. For $\sigma_{t_{src}} = 0.5$, $\Delta\sigma_{\text{seg}} = 0.25$, $\|.\| = 1$, $\kappa = 0.5$: $\sigma^* \approx 0.09$. **Within the production-sampled range, the catch-up term remains a fraction of the planned drift; it does not overwhelm.**

### (d) Pick — keep pure F2 as default, instrument flip-trigger

**Decision: keep pure F2 as default. Add an empirical flip-trigger to switch to hybrid with $\kappa = 0.5$ if SDE sampling is empirically degraded.**

Rationale:

1. **Pure F2 has bounded intra-segment error.** $|\Delta_K|$ is independent of K and bounded by $\sqrt{2\sigma_{\text{avg}}\Delta\sigma_{\text{seg}}}$. For typical segments this is comparable to the trajectory length within a segment (50–100%). This is the standard SDE-sampling regime; the deterministic-policy assumption shifts the residual error onto the loss to absorb during training.

2. **Inter-segment re-anchoring limits compound error.** The network is called at every segment boundary ([`pipeline_gmdit.py:87-91`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L87-L91)) on the *actual* deviated $x_t$, so segment-to-segment error does not accumulate. Pure F2's intra-segment 50%-trajectory-length error is bounded by segment size, not by total inference length.

3. **Hybrid is safe but not free.** The 1/σ in u-space is pipeline-cancelled by the σ-multiplication in the SDE step. But: (i) the hybrid couples u to x_t, breaking the "policy is a deterministic function of σ within a segment" property R1.1 §4.3 #3 advertised; (ii) under `FlowMapSDEScheduler` with default $h=0$ ([`flow_map_sde.py:30`](../../repos/piFlow/lakonlab/models/diffusions/schedulers/flow_map_sde.py#L30)) and `FlowEulerODEScheduler`, there is no SDE noise injected between substeps, so the hybrid degenerates to pure F2 ($\Delta = 0$). The hybrid only buys anything for stochastic samplers.

4. **Default sampler in the production pipeline is unclear.** [`pipeline_gmdit.py:23-26`](../../repos/piFlow/lakonlab/pipelines/pipeline_gmdit.py#L23-L26) annotates the scheduler as `FlowSDEScheduler | FlowEulerODEScheduler`. The deployed configs would tell us which is used in practice; I have not loaded them. If the production sampler is deterministic (FlowEulerODE), pure F2 is exactly right. If it's FlowSDE with $h=1$, the hybrid would give meaningful intra-segment error reduction.

5. **Reversible.** Switching to hybrid is a one-line addition to the policy's `pi(x_t, σ_t)` method. The decision is cheap to flip after empirical results.

**Recommended κ if we switch:** κ = 0.5. Reasoning: κ = 1 fully resets Δ per substep, which is the strongest catch-up but couples u to x_t most aggressively (and may undertrain the residual coefficients, since the loss never sees the full propagated Δ). κ = 0.5 retains half the deviation per substep, which gives meaningful damping while leaving the policy's planned trajectory the dominant velocity contribution.

**Empirical flip-trigger:** under `FlowSDEScheduler(h=1)` with default substep count, compare sample quality (FID or held-out reconstruction loss) between pure F2 and hybrid κ=0.5 at the same training checkpoint. If hybrid improves sample quality by more than ~5% on the chosen metric, switch the default. If pure F2 is within noise of hybrid, keep pure F2. Decision should be made empirically; the analytical bounds in (a)-(c) do not strongly favour either.

### (e) Compatibility with C2/Q6's anchor-velocity drift

C2 derived an anchor-velocity drift $\Delta u^{C2} = -(\pi/\sigma_{t_{src}})\sum_m m\,a_m$ at the segment anchor $\sigma = \sigma_{t_{src}}$, **independent of $x_t$**.

Hybrid F2 adds a catch-up term $\kappa(x_t - \Psi(\sigma))/\sigma$ at *every evaluation σ*. At the anchor $\sigma = \sigma_{t_{src}}$, R1.1 §5.3 derived strict anchor reproduction $\Psi(\sigma_{t_{src}}) = x_{t_{src}}$ (sin and (cos−1) both vanish at τ=0). So at the anchor, $x_t - \Psi(\sigma_{t_{src}}) = x_t - x_{t_{src}} = 0$ on-trajectory (and small even off-trajectory, since segment k starts with $x_t = x_{t_{src}}^{(k)}$ by construction). **The hybrid catch-up term vanishes at the anchor.**

So at $\sigma = \sigma_{t_{src}}$:
- Pure F2: $u = u_{src} + \Delta u^{C2}$.
- Hybrid F2: $u = u_{src} + \Delta u^{C2} + 0 = $ same as pure F2.

**The two issues are orthogonal.** C2's drift is at the anchor (single point per segment); hybrid's catch-up is at interior σ (intra-segment). Hybrid does **not** mask C2's drift, and C2's drift does **not** propagate into the hybrid catch-up. Fixing C2 (Option R: reparametrise $a_M$) and adopting the hybrid are independent decisions with non-interfering effects.

The C2 drift and the hybrid catch-up both involve $1/\sigma$-style factors but at different σ values:
- C2: $1/\sigma_{t_{src}}$ at the anchor.
- Hybrid: $1/\sigma$ at interior evaluation, with the in-tree pipeline cancelling it via $\sigma\cdot u$ in the SDE step.

Neither blowup is operationally live in the in-tree pipeline.

### (f) Confidence

- **High** confidence on (a) — the noise-std formula follows directly from [`flow_sde.py:166`](../../repos/piFlow/lakonlab/models/diffusions/schedulers/flow_sde.py#L166) and standard variance-additivity for independent gaussians.
- **High** confidence on (b)'s recursion — the propagation under pure F2 is algebraically clean.
- **Medium** confidence on (c)'s threshold for catch-up dominance. The estimate uses order-of-magnitude latent-space scales; actual production-sample $\|x_{t_{src}} - \hat x_0\|$ may differ.
- **Medium** confidence on (d). The decision is reversible; the data driving the flip is empirical (sampler-specific FID/quality).
- **High** confidence on (e) — the hybrid term vanishes at τ=0 by R1.1 §5.3 strict anchor reproduction.
