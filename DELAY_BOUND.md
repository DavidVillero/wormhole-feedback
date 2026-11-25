# The delay bound: statement and proof

Written for Piece 2 of the series and as the statement to formalise in Lean. The numbers come from `bound_zeros.py` (`results_bound_zeros.json`).

Numerical check (`delay_bound_check.py`, `results_delay_bound_check.json`): 200 other controller designs, made from random LQG weights and scaled gains. 160 stabilise, and none beats the bound. The best reach 3.9% at d = 1 (bound 8.3%) and 0.64% at d = 2 (bound 1.8%). These designs all come from the LQG family, so this is a consistency check, not a stress test. The proof below is what covers every controller.

## Setting

Sampled-data loop with sample time h = 0.25 b (one control update):

- **Plant:** the identified model of `control_lqg.build_model(d)`. One input (the pulse amplitude), two outputs (throat radius and its rate, as received at the controller). Its transfer function is P(z) = C (zI − A)⁻¹ B, a 2 × 1 rational matrix.
- **Unstable pole:** the plant has one, at p = e^{λh}, with λ = 1.182 /b the wormhole's growth rate, so |p| > 1.
- **Dead time:** P(z) = z^{−m} P̃(z) with P̃(∞) ≠ 0. The pulse and the throat's signal both travel at the speed of light (every characteristic speed in the solver is 1), so m h cannot be shorter than the light-travel time. Measured: m = 5, 8, 12, 20 samples at d = 1, 1.5, 2, 3 b.
- **Zeros both outputs share:** points z_k with |z_k| > 1 and P(z_k) = 0 in both channels. There is one at each distance (z = 1.147 at d = 1).
- **Controller:** any linear time-invariant K(z) (1 × 2) that stabilises the loop internally. Loop broken at the input: L = −K P (a scalar), input complementary sensitivity T = L/(1 + L).

## Claim

For every such controller,

    sup over |z| = 1 of |T(z)|  ≥  |p|^m · ∏_k |1 − conj(z_k) p| / |p − z_k|  ≥  |p|^m = e^{λ m h}.

So the largest plant error the loop is guaranteed to survive, 1 / sup|T|, is at most

    e^{−λ m h} · ∏_k |p − z_k| / |1 − conj(z_k) p|.

In words: a controller at distance d needs a model of the throat that is accurate to better than e^{−λ × dead time}. Each extra throat radius of distance multiplies the required accuracy by about e^{2λ} ≈ 10.6.

## Proof

1. **T is analytic outside the unit disk.** Internal stability means T has no poles with |z| ≥ 1, including at z = ∞.
2. **T(p) = 1.** L has a pole at p (the plant does, and internal stability forbids cancelling it with a controller zero). So T = L/(1 + L) → 1 there.
3. **T has a zero of order m at infinity.** L = z^{−m} × (proper), so T does too.
4. **T(z_k) = 0.** P(z_k) = 0 in both channels, so L(z_k) = −K(z_k) P(z_k) = 0. The only way out would be a controller pole at z_k, and that is an unstable pole-zero cancellation, which internal stability rules out.
5. **Map to the unit disk.** Let g(w) = T(1/w). Then g is analytic on |w| < 1, continuous up to |w| = 1, has a zero of order m at w = 0 and zeros at w_k = 1/z_k, and g(1/p) = 1.
6. **Divide out the zeros.** B(w) = w^m ∏_k (w − w_k)/(1 − conj(w_k) w) is a Blaschke product, with |B| = 1 on |w| = 1. The quotient g/B is analytic on the disk.
7. **Maximum modulus.** sup over the circle of |g| = sup over the circle of |g/B| ≥ |g(1/p) / B(1/p)| = 1/|B(1/p)|.
8. **Back to z.** 1/|B(1/p)| = |p|^m ∏_k |1 − conj(z_k) p| / |p − z_k|. Every factor in the product is ≥ 1 because |z_k| > 1 and |p| > 1, which gives the second inequality. ∎

The step from sup|T| to tolerated plant error is the small-gain theorem. A stable multiplicative error Δ at the input with sup|Δ| < 1/sup|T| cannot destabilise the loop. Some complex Δ with sup|Δ| = 1/sup|T| can.

## What it covers, and what it does not

- **Covers:** every linear time-invariant stabilising controller on this model, including designs not yet tried (H-infinity, higher order, different weights).
- **Does not cover:** nonlinear or time-varying controllers. A bound of this kind is expected to hold for them too, but it is not proved here.
- **Model, not wormhole.** It is a statement about the identified linear model. The link to the full Einstein-scalar simulation is empirical: the model's predictions match to 0.1–1% at linear order, and the calibration converges at two resolutions (NOTEBOOK.md).
- **Two-channel vs radius-only.** With both outputs, only the shared zero enters. A radius-only controller is limited further by that channel's own zeros (4.0% at d = 1 vs 8.3% with both).

## Numbers (`results_bound_zeros.json`)

| d (b) | Dead time (b) | Best possible, dead time only | With shared zero (both outputs) | Achieved by the LQG design |
|---|---|---|---|---|
| 1 | 1.25 | 22.8% | 8.3% | 4.0% |
| 1.5 | 2.0 | 9.4% | 4.9% | 1.8% |
| 2 | 3.0 | 2.9% | 1.8% | 0.65% |
| 3 | 5.0 | 0.27% | 0.20% | 0.07% |

The article's simpler form, e^{−λ(2d − 0.5)}, uses the measured first-response time (threshold 1e-3) instead of the exact dead time.

## Lean target

The analytic core is a Schwarz-lemma statement. Let f be analytic on the open unit disk with |f| ≤ M there, and suppose f has a zero of order m at 0. Then |f(w)| ≤ M |w|^m. Applied at w = 1/p with f(1/p) = 1, this gives M ≥ |p|^m, the first-order bound (steps 1–3, 5, 7). Mathlib has the Schwarz lemma (`Mathlib.Analysis.Complex.Schwarz`). The order-m version follows by dividing out w and repeating, and the Blaschke factors for z_k (step 6) are the same argument after a Möbius map. Steps 1–4 are control-theory bookkeeping: state them as hypotheses on f rather than formalising transfer functions.
