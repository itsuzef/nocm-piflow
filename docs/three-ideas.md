# Three Ideas for Modifying Pi-Flow

Yes — the cleanest way to present them is as three related ideas, with one important caveat:

> **Idea 3 is the broad framework. Ideas 1 and 2 are two concrete ways to realize it.**

So if you write them up, present them that way — not as three totally unrelated directions.

---

## The Baseline You Are Modifying

Current GMFlow-style Pi-Flow is:

$$G_\theta(x_r, r) \longrightarrow z_{\text{GMM}}$$

where $z_{\text{GMM}}$ are Gaussian-mixture parameters, and then

$$\pi_{z_{\text{GMM}}}(x_t, t) = \frac{x_t - \hat{x}_0(x_t, t)}{t},$$

with $\hat{x}_0(x_t, t)$ computed from the Gaussian-mixture policy. So the policy family is fixed: it must live inside the Gaussian-mixture parameterization.

**Your three ideas are all trying to replace or generalize that fixed policy family.**

---

## Idea 1 — Fourier / Sin-Cos Policy Parameterization

> *The idea from the whiteboard where you replace the simple path $f_s(t)$ with a sin/cos parameterization.*

### Core Idea

Instead of making the policy/path look like a Gaussian-mixture-induced posterior mean, make it a continuous function of time represented by a Fourier basis.

A clean abstract version is:

$$f(t, \xi) = \phi_0(t)\, X_{\text{start}}(\xi) + \phi_1(t)\, X_{\text{end}}(\xi) + \sum_{m=1}^{M} \left( a_m(\xi)\sin(m\pi t) + b_m(\xi)(\cos(m\pi t) - 1) \right)$$

where:
- $\xi$ is the spatial/latent coordinate,
- $\phi_0, \phi_1$ enforce the endpoint conditions,
- $a_m, b_m$ are learned coefficients.

Then the velocity is not predicted directly — it is obtained by differentiating the function:

$$\partial_t X_t(\xi) = \partial_t f(t, \xi)$$

### What the Network Outputs

$$G_\theta(x_r, r) \rightarrow \{a_m(\xi),\, b_m(\xi)\}_{m=1}^{M}$$

### Why It Is Interesting

- A truly continuous-time policy
- A compact finite representation of a continuous object
- Very cheap evaluation at arbitrary times
- A strong smoothness prior

### What It Changes Relative to GMFlow

```
Before:  network → GMM params → posterior mean → velocity
After:   network → Fourier coefficients → path function f(t) → derivative → velocity
```

The policy is no longer constrained to "look Gaussian." It is constrained to live in a basis expansion.

---

## Idea 2 — Predict PDE Parameters That Define the Function

> *Conceptually different from the Fourier idea.*

### Core Idea

Do not have the model directly predict the full velocity field $\partial_t X_t$. Instead, have it predict the parameters $\lambda$ of a PDE, and let that PDE define the whole denoising trajectory function.

In abstract form:

$$\lambda = G_\theta(x_r, r)$$

and then

$$\partial_t X(t, \xi) = F_\lambda(X, \nabla X, \nabla^2 X, t, \xi)$$

Once $\lambda$ is known, the PDE defines the function $X(t, \xi)$, and the velocity is simply the PDE right-hand side:

$$\partial_t X_t = F_\lambda(\cdots)$$

### Example Family

A possible PDE family:

$$\partial_t X = a_\lambda(\xi)\, \Delta X + b_\lambda(\xi) \cdot \nabla X + c_\lambda(\xi)\, X + f_\lambda(t, \xi)$$

Or, for the damped-oscillation flavor from the whiteboard, a second-order version:

$$\partial_{tt} X = -k_1(\xi)(X - \hat{X}_\infty(\xi)) - k_2(\xi)\,\partial_t X + u(t, \xi)$$

### What the Network Outputs

$$G_\theta(x_r, r) \rightarrow \lambda = \{a_\lambda,\, b_\lambda,\, c_\lambda,\, f_\lambda, \ldots\}$$

### Why It Is Interesting

The model is predicting not a velocity tensor, but a **dynamical law**. That gives you:
- A much stronger inductive bias
- Interpretable evolution
- Potentially better spatial coupling
- A policy/function defined implicitly as the PDE solution

### What It Changes Relative to GMFlow

```
Before:  network → GMM params → policy
After:   network → PDE coefficients → PDE-defined path/function → velocity
```

The output is no longer a probability mixture — it is an evolution rule.

---

## Idea 3 — Replace the GMM Head in Pi-Flow with a Neural Operator

> *The broadest and most ambitious idea. Closest to the note, which explicitly suggests replacing Pi-Flow's policy generator with a Neural Operator.*

### Core Idea

Right now Pi-Flow uses a fixed policy family:
- **DX:** interpolation-based denoised anchors
- **GMFlow:** Gaussian-mixture-based policy

The idea: **remove the GMM restriction and let a Neural Operator generate the policy instead.**

So instead of

$$G_\theta(x_r, r) \rightarrow z_{\text{GMM}}$$

you want something like

$$\mathcal{N}_\theta(F_r) \rightarrow m_r$$

where:
- $F_r$ is a function-like input representation,
- $m_r$ is a policy code / function code,
- a cheap query rule turns $m_r$ into the policy at later times.

A generic version:

$$\pi(x_t, t) = D_\phi(m_r, x_t, t)$$

where $D_\phi$ is a cheap decoder or evaluation rule.

### Why This Is Appealing

1. $\pi$ is **truly functional** — not forced into a Gaussian-mixture family.
2. **Greater flexibility** — the policy is no longer restricted to a fixed parametric model.

### Important Correction

Do not overstate that a Neural Operator can output arbitrarily complex continuous shapes. A Neural Operator can in principle represent much richer function families than a fixed GMM, but you still need:
- a finite representation,
- a cheap query mechanism,
- and a stable training setup.

Otherwise you have just replaced one constraint with a harder optimization problem.

A more accurate framing:

> By replacing the GMM with a Neural Operator, the policy is no longer forced to lie in a Gaussian-mixture family and can instead be learned in a richer function space, **provided we choose a representation that remains cheap to evaluate.**

---

## How the Three Ideas Fit Together

### The Hierarchy

| Role | Idea |
|---|---|
| **Umbrella framework** | Idea 3 — use a Neural Operator to replace the fixed GMM policy head |
| **Concrete instantiation A** | Idea 1 — the NO outputs Fourier coefficients |
| **Concrete instantiation B** | Idea 2 — the NO outputs PDE coefficients |

```
Idea 3: NO-generated policy (general framework)
├── Idea 1: Fourier / sin-cos parameterized policy
└── Idea 2: PDE-parameterized policy
```

---

## Polished Framing as Contributions

### Contribution A — Fourier Policy Family

We replace the fixed Gaussian-mixture policy in Pi-Flow with a boundary-constrained Fourier parameterization. The model predicts sin/cos coefficients of a continuous trajectory function, and the denoising velocity is recovered by differentiating this function with respect to time.

### Contribution B — PDE Policy Family

We replace direct velocity prediction with prediction of PDE coefficients. The network outputs the parameters of a dynamical law, and the denoising trajectory is defined implicitly as the solution of that PDE. The velocity is then obtained from the PDE right-hand side.

### Contribution C — Neural-Operator Policy Generator

More generally, we replace Pi-Flow's GMM policy head with a Neural Operator, allowing the policy to be represented in a richer function space rather than a fixed Gaussian-mixture family. This makes $\pi$ truly functional and substantially more flexible, while still preserving the Pi-Flow principle of **one expensive forward pass followed by many cheap policy evaluations**.

---

## Recommended Presentation

- **Main framework idea:** Contribution C
- **Two concrete instantiations:** Contributions A and B

The unified story becomes:

> *"We propose replacing Pi-Flow's fixed GMM head with a function-space policy generator based on Neural Operators, and we investigate two concrete parameterizations of that policy: Fourier basis expansions and PDE-defined trajectory families."*

That is a coherent research program.
