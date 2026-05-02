# The Story — What We're Doing and Why

**Layman narration. Read start to finish. No symbols unless we need them.**

---

## The problem everyone is trying to solve

Diffusion models are the reason modern image generation works. You give the model a prompt, it starts from pure random noise, and it gradually denoises that noise into an image. The denoising is done in many small steps — typically somewhere between 30 and 1000 — where at each step the network looks at the current partially-noisy image and nudges it a little closer to the clean version. It's slow, but it works, and the quality is extraordinary.

The cost is that slowness. Every step is a full pass through a large neural network. Thirty steps mean thirty forward passes. For research that's fine, but for production — anything real-time, anything on-device, anything where users are waiting — it's a showstopper. **So the central question in the field is: how do we make this faster without wrecking the quality?**

There are two schools of thought about how to attack it. The first says: use fewer steps. Teach a smaller, smarter model to jump further along the trajectory in a single pass. That's the consistency models family — distill a multi-step teacher into a one-or-two-step student. The second says: keep the steps, but make each step cheap. Don't run the expensive network at every step; run it once, and use the output to cheaply interpolate between steps.

**PiFlow lives in the second camp.**

---

## What PiFlow actually does

PiFlow's insight is that you don't need to evaluate the heavy network at every point along the denoising trajectory. You can evaluate it once at some anchor time, and have it output a *description of the trajectory itself* — a function that tells you what the image looks like at any time between now and the clean endpoint. Once you have that function, rolling forward through a hundred sub-steps is cheap, because you're just evaluating a small formula, not running a network.

The thing the network outputs is called the **policy**. You can think of the policy as a mini-map of the trajectory — "from here to the clean image, here's the path, and you can read off any point along it without calling me again." The network is expensive; the policy, once you have it, is free to query.

The catch is that the policy has to be expressive enough to actually capture the trajectory. If the policy is too simple — say, a straight line — it can't represent the curves that real denoising trajectories take, and the quality suffers. If the policy is too complex, predicting it becomes as hard as predicting the image directly, and we're back where we started.

**So the whole design question for PiFlow is: what's the right parametric family for the policy?** What kind of function should the network output such that it's simple enough to predict reliably, but expressive enough to trace real denoising trajectories?

The current PiFlow implementation uses a Gaussian mixture — the network outputs K Gaussian components (means, variances, and weights), and the trajectory is reconstructed from that. It works, but it has two problems. First, it's fixed at training time — the number of Gaussians, the interpolation schedule, the way they combine — all baked in. Second, it's not obviously the *right* family. It's an engineering choice that happens to work, not a principled one.

That's where our work starts.

---

## The three directions — and the hierarchy

The paper's Section 7 lays out three ideas for replacing the Gaussian mixture head with something better. As written, the three come across as alternatives. Our reading is that they're not alternatives — they form a hierarchy, and that hierarchy is the real story:

- **Idea 3 (Neural Operator as policy generator) is the umbrella.** The network stops outputting a fixed-shape tensor and starts outputting a *function*. A neural operator is exactly the right object for this — it's a class of networks designed to produce functions as output.
- **Idea 1 (Fourier policy) is one instantiation.** The function the operator outputs is a Fourier series — a sum of sines and cosines with learned coefficients.
- **Idea 2 (PDE policy) is the other instantiation.** The function the operator outputs is the solution to a partial differential equation, where the operator predicts the PDE's coefficients.

Reframed this way, the thesis becomes clean: *we turn the policy head into a function generator, and we investigate two concrete ways of parameterizing that function — basis expansion and differential equation.* That's a stronger story than "here are three things we might try."

Let's walk through each instantiation and what's actually hard about it.

---

## Idea 1 — the Fourier policy (and the trig schedule problem)

The Fourier idea is the simplest to state. Instead of predicting a Gaussian mixture, the network predicts the coefficients of a sine-and-cosine expansion that describes the trajectory. You give the expansion enough terms to be expressive, and the boundary conditions pin it to the correct start and end images. The network's job is to predict how the trajectory curves between those endpoints.

Here's where it gets interesting. The current PiFlow code uses what's called a **linear interpolation schedule** — at time `t` along the trajectory, the image is `(1-t) × clean + t × noise`, linearly mixed. This is simple and it works, but it has a quiet cost: it's what's called a *variance-exploding* process. As you add more noise, the total variance of the image grows unboundedly. That's fine mathematically but has some practical consequences — particularly at the clean end of the trajectory, where the signal-to-noise ratio does awkward things.

An alternative is the **trig interpolation schedule**. At time `t`, the image is `cos(πt/2) × clean + sin(πt/2) × noise`. Same endpoints — clean at `t=0`, pure noise at `t=1` — but because of the trig identity `cos² + sin² = 1`, the total variance stays constant along the trajectory. This is called a *variance-preserving* process. It's used in a lot of modern diffusion work (TrigFlow, EDM, sCM) because it gives cleaner mathematics and nicer numerical behavior.

**So why not just switch PiFlow to the trig schedule?** Here's the problem. PiFlow doesn't just do interpolation — it does a Bayesian posterior update every time it samples. When the model has two noisy observations of the same clean image at different times, it has to *fuse* them into a single better estimate. That fusion is a Gaussian posterior update, and it depends on the specific relationship between the signal coefficient `α` and the noise coefficient `σ`. The current code hardcodes the linear relationship `α = 1 - σ` directly into the posterior formula. The moment you switch to trig, that formula is wrong.

This is the piece we think is missing from the paper, and the piece I want to work on first: **re-derive the posterior update so it works for an arbitrary schedule.** The plan is to express the posterior in information form — precision gain `ζ`, information vector `ν`, component denominators, updated means, updated weights — such that plugging in any valid `(α, σ)` pair gives the right answer. Under the linear schedule it should reduce exactly to the existing code (that's the sanity check). Under the trig schedule it gives us the new formulas we need. Verify it symbolically, verify it numerically, then use it as the foundation for everything else.

Without this piece, any attempt to prototype the trig Fourier policy is broken from step one.

There's also a numerical story around the trig schedule worth flagging. Under linear, the signal-to-noise ratio has a gentle slope. Under trig, it behaves like `cot²(πt/2)` — which blows up near `t = 0`. The existing numerical guards in the code (small-epsilon clamping) were designed for the linear regime and probably aren't sufficient for trig. Similarly, the Euler integration steps are currently uniform in `σ`-space, which works for linear because `σ` is linear in `t`. Under trig, the effective step size in image-space varies by a factor of π/2 across the trajectory. That might or might not matter in practice, but we should know before we're chasing mysterious training instabilities later.

---

## Idea 2 — the PDE policy

The PDE idea is more elegant, and also more uncertain.

Instead of describing the trajectory as a sum of sine waves, we describe it as the solution to a differential equation. The network's output isn't a list of coefficients for basis functions — it's a list of coefficients for the equation itself. Predict the equation, let the math solve the trajectory for you.

The specific form the paper considers is a damped oscillator: the trajectory obeys an equation of the shape "second derivative equals a restoring term toward an equilibrium, plus a damping term, plus a forcing term." The network predicts the restoring strength `k_1`, the damping strength `k_2`, and the forcing function `f`. The trajectory that comes out of this equation is a physically-motivated curve — it can oscillate, decay, overshoot — and the space of achievable trajectories is parameterized by just those three numbers (or three functions, if we let them vary across the image).

**Why this is appealing:** it's mathematically crisp. The equation has a closed-form solution for any given set of coefficients, which means evaluating the policy at any time is a pure formula — no neural network, no iterative solver, just arithmetic. That's potentially even cheaper than the Fourier approach, and it comes with inductive bias — the ODE structure encodes that trajectories should be smooth, should respect conservation laws, should behave physically.

**Why it's hard:** two problems, neither of them solved yet.

The first is **expressiveness**. If `k_1`, `k_2`, and `f` are scalars — single numbers shared across every pixel — then every pixel in the image is running the *same* trajectory. That can't be right. A 128×128 image has structure; neighboring pixels should have correlated but different trajectories. So the coefficients probably need to be spatial maps — `k_1(ξ)` varies across the image. But now we've got output dimensionality close to the image itself, which brings us into the same territory as the Fourier approach. The distinction between Idea 1 and Idea 2 starts to blur: a PDE with fully spatial coefficients and a Fourier expansion with position-dependent coefficients might be mathematically equivalent, or near-equivalent. We don't actually know where the distinction lives.

The second is **stability**. The damped oscillator equation has a closed-form solution only when the coefficients are in a valid range. If `k_1` comes out positive — meaning "repelling from equilibrium" instead of "restoring toward equilibrium" — the trajectory diverges to infinity. We'd need to constrain the network's output so it can never emit unstable coefficients. A softplus activation would do it, but then we've restricted the class of trajectories the policy can represent. The paper doesn't address this and I don't know yet whether the restriction matters.

**My read: Idea 2 isn't ready to prototype yet.** The expressiveness question and the stability question both need answers before we commit to writing code. Idea 1 is the one with the cleaner path to a first experiment.

---

## Idea 3 — the neural operator umbrella

Idea 3 is the framing, not a separate experiment.

A neural operator is a network that maps functions to functions. Standard networks take a fixed-size input (like an image) and produce a fixed-size output (like a label or another image). A neural operator takes a function as input — defined on some continuous domain — and produces a function as output. The classical example is the Fourier Neural Operator (FNO), which processes its inputs in frequency space and is known for learning resolution-independent solution operators to PDEs.

In our context, the input is the current noisy image treated as a function of spatial location, and the output is the policy treated as a function of time. The neural operator is the right object because the thing we're predicting (the policy) *is a function*, not a fixed tensor. And because it's a function, we can decide after the fact how to represent it — as Fourier coefficients, as PDE coefficients, or as something else.

**This is the unifying move.** Ideas 1 and 2 aren't alternatives to Idea 3 — they're different choices about *what basis the neural operator outputs in.* Idea 3 is the architecture; Ideas 1 and 2 are the representations.

The practical work on Idea 3 comes later. We'd need to take the FNO implementation from the `neuraloperator` repo and figure out how to plug it into PiFlow's existing `(image, time) → policy` contract. That's an engineering task I haven't started. The main thing we get from Idea 3 *now* is the paper framing — it's the umbrella under which everything else makes sense.

---

## Where we are and what's next

Nothing has been prototyped yet. No math has been formally derived yet. This is a planning stage — we're aligning on the approach before committing engineering time.

**The sequence I'd propose is:**

1. **Derive the schedule-agnostic posterior.** Pure math, no training runs, no infrastructure. Unblocks everything else. Check reduction to the existing code under linear. Verify under trig symbolically and numerically.
2. **Prototype Idea 1 (trig Fourier policy).** Smallest code delta. Clear win criterion against the linear-GMM baseline. Probably a week to first numbers.
3. **Resolve the Idea 2 expressiveness and stability questions** before any prototyping. Decide whether it's a meaningfully different approach from Idea 1 or a near-equivalent re-parameterization.
4. **Frame the paper around Idea 3** once something is actually working. The umbrella story only lands if we have at least one instantiation with numbers behind it.

The whole program is bounded by one honest constraint: we're trying to make PiFlow faster or better (ideally both) by giving the policy head a more expressive, more principled structure. If the trig Fourier prototype doesn't beat the linear-GMM baseline, we learn something. If it does, we've got a clean story — change the interpolation schedule, change the policy family, re-derive the posterior — each piece is a self-contained contribution.

That's the whole arc.
