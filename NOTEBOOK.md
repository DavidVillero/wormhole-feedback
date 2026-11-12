# Gravity step: Einstein–scalar wormhole solver

Lab notebook, kept as written during the work (2026-11-09 to 2026-11-11). Newest results are at the bottom.

This is the first code in the project that evolves spacetime geometry together with its matter source. It reproduces a published result (the instability of the Ellis–Bronnikov wormhole). It contains no new physics and no FTL solution.

## What it solves

Spherical symmetry, conformal gauge, G = c = 1, lengths in units of the throat radius b:

    ds² = e^{2σ(t,x)} (−dt² + dx²) + R(t,x)² dΩ²

Matter is a massless scalar with stress tensor ε[∂φ∂φ − ½g(∂φ)²]. ε = −1 is a phantom (ghost) field, ε = +1 a canonical one. This phantom field is new to the project: the earlier flat-space work used a canonical scalar only.

Evolved: R, σ, φ (three wave equations, all characteristic speeds exactly 1). Monitored but not enforced: Hamiltonian and momentum constraints H, M. Method: second-order centred differences, RK4, Kreiss–Oliger dissipation, grid |x| ≤ 40b with the outer boundary causally disconnected from the analysed region |x| ≤ 10b.

Initial data: time-symmetric, R = √(x² + b²), σ = amp·exp(−x²), and φ′ fixed by solving the Hamiltonian constraint exactly. amp = 0 is the exact static wormhole.

## Checks and results

| Check | Result |
|---|---|
| Equations vs Einstein tensor (`derive_check.py`, sympy) | Every component of G − 8πT is a combination of the solver's residuals; Ellis–Bronnikov satisfies all five exactly |
| Constraint convergence (dx = 0.05 → 0.00625) | H and M fall by 4.00–4.03 per halving |
| Self-convergence of throat radius | Ratios 4.02–4.11 (second order) |
| Unstable-mode growth rate | 1.185 /b measured; published 1/0.846 = 1.182 /b (González, Guzmán & Sarbach 2009). Difference 0.2% |
| amp > 0 | Collapse: trapped surfaces form, areal radius → 0, apparent-horizon radius ≈ 0.65–0.67 b when the run stops |
| amp < 0 | Expansion: throat radius grows without bound for the whole run (to 82 b at t = 120 b) |
| amp = 0 | Stays static only until truncation error seeds the unstable mode; the drift shrinks 4× per halving of dx, as it should |
| Canonical scalar (ε = +1) | No time-symmetric throat data exists: the constraint demands φ′² < 0 |

Numbers are in `results.json`, the figure is `validation.png`.

## Caveats

- The growth-rate fit uses the half-difference of the ±0.0001 runs over 3 ≤ t ≤ 6, which cancels the truncation-seeded part. Other windows and amplitudes give 1.185–1.194.
- Collapse runs stop when R < 0.02 b anywhere. The horizon radius quoted is the last value before that, differs by up to 2.5% between the two finest grids, and is not a settled final black-hole mass. The published collapse runs used different perturbations, so their horizon radii (0.45–0.53) are not directly comparable.
- In the expansion runs the throat's proper time appears to approach a finite value (~7.5 b) while R keeps growing. Constraints stay below 3×10⁻⁵, but this was not checked against the literature or at a second resolution. Treat it as an observation.
- Spherical symmetry only. Minimal coupling only (no ξRφ² term yet). Classical field only.

## Reproduce

    python3 derive_check.py      # symbolic verification, ~1 min
    python3 run_suite.py cmfx    # 28 runs, ~30 s on 4 cores
    python3 analyze.py           # results.json, validation.png

Needs numpy, matplotlib, sympy. `minus_long.json` came from `run(dx=0.025, amp=-0.01, T=120, L=135, out_dt=0.5, snap_dt=20)`. `runs/test.json` is a leftover from the first smoke test (before an initial-data fix) and can be deleted.

## Sources

- González, Guzmán & Sarbach, linear analysis: https://arxiv.org/abs/0806.0608
- González, Guzmán & Sarbach, nonlinear evolution: https://arxiv.org/abs/0806.1370
- Shinkai & Hayward 2002: https://arxiv.org/abs/gr-qc/0205041

---

# Control study, steps 1–3 (added 2026-11-09)

Question: can a controller hold the wormhole open, and at what cost? Open-loop (hand-tuned) pulses are prior art (Shinkai & Hayward 2002; Xu, Chew & Yeom 2025). Closed-loop feedback is the target. These steps build the tools and the open-loop baseline; no feedback controller has been run yet.

## Actuator (`control_solver.py`)

Two scalars: phantom (supports the wormhole) and normal. No source terms are used. A control pulse is written into the scalar fields on a slice and the Hamiltonian and momentum constraints are re-solved outward for (R, R_t), so every slice is valid Einstein data and the constraint monitors stay meaningful.

- Phantom pulses that are odd in x couple linearly to the unstable mode: one sign pushes toward expansion, the other toward collapse. One actuator therefore corrects both directions.
- Normal pulses only push toward collapse.

## Open-loop baseline (`control_steps123.py open`, `balance`)

| Case | Untuned lifetime | Best tuned lifetime | Gain per decade of tuning |
|---|---|---|---|
| Normal "traveller" pulse (amp 0.02 from x = 6) + one phantom pulse from x = 7 | 6.5 b | 37.4 b | 1.67 b |
| Static wormhole, trimmed against grid error | 7.4 b | 37.0 b | 1.92 b |

Both were tuned to double-precision round-off (bracket ~1e-16 relative), so ~37 b is the ceiling for a single tuned parameter at this precision. The predicted gain is ln(10)/1.185 = 1.94 b per decade; the static case matches to 1%. The traveller case is faster because the tuned wormhole is left with a smaller throat (radius ≈ 0.86 b and still shrinking at t = 30), and the instability rate scales inversely with throat radius.

## Unstable mode and actuator gain (`control_steps123.py gain`)

- Mode shape saved in `unstable_mode.npz`; profiles 2 b apart agree to 0.6% within |x| < 3 b.
- Growth rate from pulse responses: 1.181–1.182 /b at all four distances.
- Throat response per unit pulse field amplitude, referred to arrival time: 39, 74, 128, 200 for launch distances 3, 5, 8, 12 b (roughly proportional to distance, from spherical focusing).
- After a pulse arrives, throat readings track the pure growing mode to 2% within about 1–1.6 b.

## Caveats

- One resolution only (dx = 0.025). Critical amplitudes are specific to this grid.
- `pulse_mass_per_unit_amp` in the JSON is an interference term with the background field, linear in amplitude and sign-dependent. It is not a usable control-cost measure; choosing one is open.
- Symmetric (both-sides) pulses only.

Figure: `control_baseline.png`. Numbers: `results_control.json`.

---

# Control study, step 4: closed-loop feedback (2026-11-09) — NOT YET WORKING

`control_feedback.py` implements a delayed, model-based feedback loop: read the throat radius and its rate with light-travel delay d, predict the unstable-mode amplitude at the pulse's arrival (including pulses in flight, from the measured impulse response), launch a phantom pulse to cancel a fraction of it.

## Result

No configuration stabilised the wormhole. Lifetimes in throat radii (`results_step4.json`):

| Run | Lifetime |
|---|---|
| No control, raw wormhole | 7.4 |
| Controller at d = 1, estimator 2, raw wormhole | 23.9 (best from a raw start) |
| Controller at d = 1, estimator 1, raw | 14.4 |
| Controller at d = 2, estimator 2, raw | 16.4 |
| Controller at d = 3, estimator 2, raw | 11.2 |
| Grid-balanced wormhole, no control (step 2) | 37.0 |
| Grid-balanced wormhole + controller at d = 1 | 16.5–25.9 (the controller makes it worse) |

So feedback extends a raw wormhole's life about threefold, but does not beat a single open-loop tuning to round-off, and it destabilises an already-balanced wormhole.

## What was established

- The plant model is right at linear order: a three-pulse train injected mid-run matches the superposed impulse responses to 0.1–1%.
- The control law is stable on the single-mode linear model (offline simulation with the measured impulse response).
- One real bug was found and fixed: the pulse bookkeeping differenced two exponentially growing terms with a small gain mismatch. The hand-over is now continuous.
- Failure pattern (`control_step4_diagnostic.png`): the throat radius is held within ~1% for 10–12 b, while the pulse amplitudes and the conformal factor at the throat both grow exponentially (rate ≈ 0.5–0.6 /b), until the pulses saturate.

## What was not established

The cause. The working hypothesis is that a single growing mode is not the whole plant: pulses also shift the wormhole along its family of static solutions (a neutral mode, seen in step 2 as the throat settling at 0.86 b), and the estimator misreads that. A first attempt to separate the two (slow filter on the neutral part, options `tau_z`, `kz`) did not help, so the hypothesis is unconfirmed. Other candidates not yet tested: drift of the lapse at the throat (the controller's growth rate is per coordinate time, not proper time), and estimator transients during pulse passage.

## Next

System identification before more controller design: fit a small linear state-space model (growing, decaying, neutral, plus the throat lapse) to pulse-response data, check it predicts the closed-loop failure, then design the regulator on that model.

---

# Control study: system identification and redesigned regulator (2026-11-09)

Status: the cause of the step-4 failure is identified and confirmed. The redesigned regulator holds a raw wormhole for 46–51 b (no control: 7.4; step 4: 23.9; open-loop tuning ceiling: 37). It is still not stable indefinitely.

## Identification (`sysid.py`, `control_regulator.py`)

Data: throat response to one phantom pulse pair from d = 1 b (amplitude 1e-9, odd part of a ± pair on the grid-balanced wormhole), sampled every 0.05 b for 13 b. Usable to about 7 b after arrival; beyond that the growing mode's own nonlinearity exceeds the terms being fitted.

Per unit pulse amplitude, the throat radius responds with:

| Term | Value | Comment |
|---|---|---|
| Growing mode | 3.03 · e^{1.1819 s} | rate matches the published 1.182 |
| Neutral shift | ≈ 2.9 at s = 3, still rising slowly | the pulse leaves the wormhole on a neighbouring static solution |
| Transient | measured | gone by s ≈ 2 |

The neutral shift is as large as the growing-mode kick. The step-4 controller assumed it was zero, read it as growing mode, and multiplied it by the loop-delay growth factor e^{2·lam·d} ≈ 10.6 at every step.

An ERA state-space fit (in `sysid.py fit`) recovers the growing rate 1.182 at every model order but does not give stable estimates of the slow modes (they move between −0.1 and +0.2 with order), so the regulator uses the structured model above.

## Confirmation

The step-4 controller run against the two-term model (growing mode + constant shift + measured transient, driven by the recorded uncontrolled baseline) fails the same way as in the full simulation: lifetime 17.2 b vs 16.5 b, with pulse amplitudes within 15% rms up to saturation (`control_sysid.png`, middle panel).

## Redesigned regulator

Same predictor structure, but each launched pulse's known non-growing response (transient, then fitted neutral tail) is subtracted from the estimator reading before extrapolating across the delay. Raw wormhole, controller at d = 1 b:

| Variant | Lifetime |
|---|---|
| Estimator 2 (rejects constant offsets) | 46.2 b |
| Estimator 1 | 27.7 b |
| Estimator 2 + small feedback on measured radius and its rate | 50.9 b |

## Why it still fails

The surviving failure is a slow oscillation of the throat radius (period ≈ 24 b, amplitude growing about 6× per half period). The neutral response has a slow tail that 6–7 b of clean data cannot resolve: the fitted relaxation rate sits at the lower bound of the search (0.05 /b), and its asymptote is an extrapolation. A scan of twelve hand-tuned radius-feedback gains moved the lifetime between 31 and 51 b without stabilising it.

## Limits

- One controller distance (d = 1 b), one resolution (dx = 0.025), symmetric pulses.
- The 15% agreement in the confirmation is sensitive to the fitted gain; an earlier fit of the same model agreed within 1% up to t = 8 b.
- No delay-bound sweep (step 5) yet; it needs a regulator that holds first.

## Next

Measure the neutral response over a long window directly, by cancelling the growing mode physically (a second pulse tuned by section search, as in step 2) and recording 30 b of what remains. Then fit the slow sector and redo the regulator with it.

---

# Control study: long-window tail measurement and a regulator that holds (2026-11-09)

Status: with the slow tail measured and modelled, the regulator holds a raw wormhole for the full 225 b tested (no control: 7.4 b; open-loop ceiling: 37 b; previous regulator: 51 b). Not established: unbounded stability, resolution independence, or robustness to anything but small disturbances.

## Tail measurement (`sysid_pair.py`)

A pulse (amplitude 1e-6) at t = 0 and a cancelling pulse at t = 1, with the second amplitude tuned by section search to round-off, for both signs. Tuned ratios: 3.26142 and 3.26026 (e^lam = 3.26088). The odd part gives 17 b of clean data on the non-growing response.

The neutral throat shift of one pulse is a power law, not an exponential:

    n(s) = 5.79 − 5.51 (s + 2.6)^(−0.375)      per unit pulse amplitude, fit rms 0.0024 over 4 ≤ t ≤ 17

So the shift is 2.9 at s = 3 and is still approaching 5.8 very slowly. This is why the earlier exponential fit sat on its bound. The asymptote 5.79 is an extrapolation.

## Regulator (`control_regulator.py`, `control_long_run.py`)

Same predictor as before with the power-law tail in the pulse bookkeeping, estimator 2, and a small growing-mode set-point proportional to the modelled radius shift (gz = 0.05). Controller at d = 1 b, update every 0.25 b, loop delay 2 b (growth factor 10.6 across the delay).

| | |
|---|---|
| Outcome | alive at t = 225 b (proper time 225.2 b), run ended by choice |
| Largest throat-radius offset | 1.5% |
| Radius drift after t = 100 | +4.5e-5 per b, linear, not levelling off |
| Pulse amplitude | peak 1.3e-4; rms falls from 2e-5 (first 25 b) to 9e-8 (last 25 b) |
| Constraints (rms, \|x\| < 10 b) | H ≤ 3.3e-5, M ≤ 1.3e-5, i.e. truncation level throughout |

Sensitivity to gz: 0 expands at 133 b; 0.05 and 0.1 hold to 150 b; 0.15 collapses at 144 b; 0.2 expands at 92 b. The stable window is narrow.

## Disturbance test

A normal-matter pulse from x = 6 (width 1) on top of the raw wormhole, regulator as above:

| Pulse amplitude | Result |
|---|---|
| 0.0005 | held for the 60 b tested, radius offset ≤ 0.7% |
| 0.002 | lost at 17.5 b |
| 0.005 – 0.02 | lost within 7–11 b |

The controller has no model of an incoming pulse's own non-growing signature, so it misreads it the same way step 4 misread its own pulses.

## Caveats

- The linear radius drift is close to the uncontrolled grid-balanced baseline's drift (~3e-5 per b), which suggests truncation error, but this was not checked at a second resolution. If it is physical the run would eventually leave the linear regime.
- One resolution (dx = 0.025), one controller distance (d = 1 b), symmetric pulses, spherical symmetry. The model and the plant are the same discretisation, so this shows the controller works on this numerical system.
- Phantom pulses of both signs are assumed available on demand. Nothing here makes the exotic matter itself more plausible.
- A 300 b run was attempted in the background and was killed by the environment before finishing; 225 b is the longest completed run.

Figure: `control_regulator.png`. Numbers: `results_regulator.json`.

---

# Resolution check (2026-11-09)

Status: passed. The whole chain (re-balance, single-pulse response, tuned pulse pairs of both signs, refit, 225 b closed-loop run) was repeated at dx = 0.0125. The radius drift, the control effort and the constraint residuals all fall by a factor of 4, as second-order grid error should. The drift seen at dx = 0.025 is numerical.

| Quantity | dx = 0.025 | dx = 0.0125 | Ratio |
|---|---|---|---|
| Outcome at t = 225 b | alive | alive | |
| Throat-radius offset at t = 225 | 1.46% | 0.37% | 3.96 |
| Radius drift after t = 100 (per b) | 4.54e-5 | 1.16e-5 | 3.92 |
| Peak pulse amplitude | 1.28e-4 | 3.19e-5 | 4.01 |
| rms pulse amplitude, last 25 b | 9.0e-8 | 2.5e-8 | 3.6 |
| Max H residual | 3.27e-5 | 8.09e-6 | 4.05 |
| Trim needed to balance the static wormhole | −1.961e-4 | −4.902e-5 | 4.0007 |

Calibration is resolution-independent: growth rate 1.18200 vs 1.18204, growing gain 2.9861 vs 2.9861, cancelling ratio 3.260837 vs 3.260803, and the 17 b pair response differs by at most 0.004 on a typical size of 7.8 (0.05%).

What the check does not settle:

- The power-law tail's exponent and asymptote are poorly determined. The two resolutions give (exponent, asymptote) = (0.375, 5.79) and (0.325, 6.07), yet the fitted curves agree to 0.1% over the measured range. Only the curve inside 3 ≤ s ≤ 16 is established.
- Stability beyond 225 b, other controller distances, disturbance rejection, and non-symmetric pulses are unchanged from the caveats above.
- The same gain (gz = 0.05) was used at both resolutions; the width of its stable window was not re-scanned on the fine grid.

Scripts: `rescheck.py` (resumable stages), `rescheck_report.py`. Numbers: `results_rescheck.json`. Figure: `control_rescheck.png`.

---

# Distance sweep (2026-11-10)

Status: the regulator holds at d = 1 b and fails at every larger distance tried. The failure distance for this design lies between 1 and 1.5 b. Whether that is a property of the wormhole or of this regulator is not established.

Each distance was calibrated from scratch (single-pulse response, cancelling pairs of both signs tuned to round-off, power-law tail fit), with the same pulse width (0.3 b), at dx = 0.025.

Two changes were needed before the sweep was meaningful, and both apply to d = 1 as well:

- Estimator order 3. Order 2 reads the grid's steady radius drift as growing mode; that misreading is multiplied by the delay growth factor, which is harmless at d = 1 (×10.6) and fatal at d = 2 (×113).
- Readings before t = 10 b are ignored. The balanced initial data has its own start-up transient, which every estimator misreads.

Start: grid-balanced wormhole (a raw one collapses at 7.4 b, before a loop with delay ≥ 4 b can act). No radius set-point. `sweep.py`, `sweep_report.py`, `results_sweep.json`, `control_sweep.png`.

| d (b) | Loop delay (b) | Growth across delay | Lifetime, gain 0.5 | Lifetime, gain 0.35 |
|---|---|---|---|---|
| 1 | 2 | 10.6 | open at 150 (run end) | open at 150 (run end) |
| 1.5 | 3 | 34.7 | 127 | 164 |
| 2 | 4 | 113 | 71 | 91 |
| 3 | 6 | 1202 | 32 | 37 |
| no control | | | 20–37 (round-off dependent) | |

Other gains at d = 2: 0.3 → 104 b, 0.4 → 82 b, 0.75 → 54 b; 0.1 and 0.05 → 27–28 b. A gain below about 0.26 cannot work at any distance, because the mode grows by 1.34 per 0.25 b update.

## How it fails

At d ≥ 1.5 the throat is held with pulses near 1e-6 for tens of b, then a fast oscillation in the pulse sign (period about two updates) grows exponentially until the pulses saturate. Its growth rate rises with the gain (d = 2: roughly 0.37 /b at gain 0.35–0.4, 0.46 at 0.5, 0.53 at 0.75) and the onset times are consistent with exponential growth from round-off level. That points to a closed-loop instability of the regulator on its own model error, not to an external disturbance. This reading is inferred from timing; the seed was not measured directly.

## What is and is not shown

- Shown: sensitivity to any non-growing signal the estimator misreads scales with the delay growth factor e^{2·lam·d}; two specific misreadings (steady drift, start-up transient) were identified and removed.
- Not shown: a fundamental delay bound. A perfect model would allow any delay in principle. The limit found here is where this regulator's model error, multiplied by e^{2·lam·d}, exceeds its stability margin. A better model or estimator could move it.
- d = 1 is not shown to be stable. In these order-3 runs the control effort falls until t ≈ 50 b and then rises about tenfold by t = 150 b (still below 1e-6). That is either a very slow instability (rate of order 0.02–0.03 /b) or the loop tracking the grid's radius drift; the runs are too short to tell. The earlier 225 b runs (order 2 with a radius set-point) showed effort still falling at the end.
- The power-law tail fit is poorly conditioned at d = 3 (one parameter on its bound).
- One resolution, one pulse width, symmetric pulses.

## Next options

1. Find the model error that drives the fast oscillation (candidates: the hand-over from measured kernel to analytic tail at 2 b after arrival; sampling of the third-derivative estimator; radius drift moving the plant off its calibration point).
2. Replace the hand-built predictor with a standard observer-based design on the identified model, with an explicit robustness margin.
3. Repeat d = 1.5 and 2 on the fine grid to see whether the oscillation's growth rate depends on resolution.

---

# Hunting the fast oscillation, option 1: model error (2026-11-10)

Status: the leading suspect was a real error but not the cause. The cause is still not identified. `results_option1.json`.

## Suspect 1: hand-over from measured kernel to fitted tail — tested, rejected

The power-law tail was fitted to the throat radius. After the estimator's derivatives it is wrong between s = 2 and 5: the error is 0.38 of the loop margin at d = 2 with estimator 3 (0.06 at d = 1).

Fix: the non-growing kernel is now built from the pair data alone by backward recursion, ng(s − 1) = (ng(s) − q(s))/beta, starting from the tail at s = 13. Errors shrink by 1/beta per step and neither the growth rate nor the growing gain is needed to high precision (`ng_table` in `control_regulator.py`; now the default, `WH_KERNEL=old` restores the previous one).

Result with the corrected kernel, estimator 3, gain 0.5 / 0.35:

| d | Before | After |
|---|---|---|
| 1.5 | 127 / 164 | 130 / open at 150 |
| 2 | 71 / 91 | 74 / 91 |

Unchanged within a few b, so this error was not driving the oscillation.

A dead end worth recording: using the measured single-pulse response out to s = 6 (subtracting k·e^{lam·s}) made things worse (d = 2: 42–48 b). It needs the growth rate to about 1e-6, and the four calibrations give 1.18197–1.18202.

## Suspect 3: radius drift moving the plant off its calibration — suggestive, unconfirmed

Estimator 3 has transient spikes about ten times the growing gain while a pulse passes, so a 0.1% change in the plant's response is a large loop error at d = 2. The grid's radius drift provides such a change. At the onset of the oscillation, (radius offset) × (delay growth) is 0.12–0.15 at d = 1.5 and 0.24–0.33 at d = 2, and at d = 3 it is above 2 before control starts. The ordering fits; the numbers are not a clean threshold. Changing when control starts did not separate this from a fixed-rate instability growing from round-off (starting later is not possible: the balanced wormhole is already failing by t ≈ 20 on this grid).

## Suspect 2: estimator sampling — not tested separately

## What would decide it

The fine grid has four times less drift. If drift is the trigger, the onset at d = 1.5 and 2 moves about four times later there; if not, it stays put. That is option 3, and it now has a sharp prediction.

---

# Option 3: the fast oscillation on the fine grid (2026-11-10)

Status: the oscillation depends on resolution. Halving the grid spacing doubles the time it takes to appear, at both distances. That rules out a fixed-rate instability of the regulator and is what a drift-driven instability with growth rate proportional to the radius offset predicts. It is not yet proof that drift is the cause: that needs a run with the radius held fixed.

Distances 1.5 and 2 were recalibrated from scratch at dx = 0.0125 (drift four times smaller) and rerun with the same regulator (estimator 3, pair-built kernel, no radius set-point). `results_option3.json`, `control_option3.png`.

| d | Gain | Lifetime, dx = 0.025 | Lifetime, dx = 0.0125 | Ratio |
|---|---|---|---|---|
| 2 | 0.5 | 74 | 131 | 1.77 |
| 2 | 0.35 | 91 | 168 | 1.84 |
| 1.5 | 0.5 | 130 | 263 | 2.02 |
| 1.5 | 0.35 | open at 150 | open at 300 | |

Three predictions were on the table: ×4 (oscillation starts at a fixed radius offset), ×1 (drift irrelevant), ×2 (growth rate proportional to the offset, so log-amplitude grows as drift·t²/2 and time scales as drift^(−1/2)). The data pick ×2.

The sharper check: when the fast (sign-flipping) pulse amplitude reaches 1e-5,

| d | Gain | Time, coarse → fine | Radius offset, coarse → fine | drift·t², coarse vs fine |
|---|---|---|---|---|
| 2 | 0.5 | 52 → 102 | 2.33e-3 → 1.16e-3 | 0.122 vs 0.120 |
| 2 | 0.35 | 67 → 127 | 3.15e-3 → 1.50e-3 | 0.202 vs 0.186 |
| 1.5 | 0.5 | 106 → 216 | 4.76e-3 → 2.48e-3 | 0.513 vs 0.542 |

Twice the time at half the offset, with drift·t² matching to 1–8%.

One thing does not fit the simple law. Early on, before any visible growth, the fast amplitude sits about ten times higher on the fine grid than on the coarse one at d = 1.5 (`control_option3.png`, middle panel). With different starting levels, equal drift·t² would imply different proportionality constants on the two grids. So the ×2 in time is solid; the reading "rate proportional to offset, same constant" is not.

Consequences:

- If the drift reading is right, lifetime at these distances scales as 1/dx at fixed gain, and the failures at d = 1.5 and 2 are a grid effect acting on a drift-sensitive regulator, not a delay bound. Two grids cannot establish a scaling law.
- The sensitivity is roughly four times larger at d = 2 than at d = 1.5 (drift·t² of 0.12 vs 0.51), close to the ratio of delay growth factors (3.3).
- By the same scaling d = 1 with estimator 3 should fail too, somewhere beyond the 150 b tested on the coarse grid. Not checked.
- Calibration is again resolution-independent (d = 2 growing gain 6.4479 vs 6.4477).

What this hands to option 2: the regulator has to hold the throat radius at its calibration value (or be made insensitive to it). The current one regulates only the growing mode and lets the radius wander.

Not established: the proportionality constant per distance and gain (direct fits were too noisy to report), and what happens once the radius is held.

---

# Option 2: observer-based regulator with an explicit margin (2026-11-10)

Status: the observer-based design does better than the hand-built predictor at every distance tested, holds the throat radius instead of letting it wander, and comes with a computed robustness margin. It holds at d = 1 and d = 1.5 for the runs made and still fails at d = 2. The margin calculation gives the delay bound in a design-independent form.

`control_lqg.py`, `control_lqg_run.py`, `lqg_report.py`, `results_lqg.json`, `control_lqg.png`.

## Design

- Model (28–44 states, one per 0.25 b update): a delay line holding recent pulses with response taps taken from the pair-data kernel; one unstable state, one neutral state and four slow states (rates 0.012–0.8 /b) approximating the power-law tail; two disturbance states (constant push on the growing mode, steady radius drift). Inputs: pulse amplitude. Outputs: the throat radius offset and its rate as received, i.e. already delayed by d.
- Controller: steady-state Kalman filter plus LQR on the received readings. No estimator derivatives, no hand-over, no separate radius set-point.
- Checks on the model: growing gains in the two channels agree with rate = lam × radius to 1e-4; slow-tail fit rms 5e-4 to 1e-3.
- Tuning was chosen by scanning about 60 weight combinations at d = 1 for the best margin and then used unchanged at all distances. LQG has no guaranteed margin and the result is sensitive to the weights (margins between 0.5% and 4% across the scan).

## Robustness margin

"Tolerated plant error" is 1 / max|T|: the largest fractional error in the plant's response, at any frequency and phase, that the loop is guaranteed to survive. The loop delay is from launch to first response at the controller (2d − 0.5 b for these pulses).

| d | Loop delay | Tolerated plant error, this design | Best possible for that delay |
|---|---|---|---|
| 1 | 1.5 b | 4.0% | 17% |
| 1.5 | 2.5 b | 1.8% | 5.2% |
| 2 | 3.5 b | 0.65% | 1.6% |
| 3 | 5.5 b | 0.07% | 0.15% |

"Best possible" is exp(−lam × delay). It follows from two facts: the complementary sensitivity T must equal 1 at the unstable pole, and a delay is an all-pass factor of T, so by the maximum-modulus principle max|T| ≥ exp(lam × delay) for any stabilising controller. This is a standard result in the fundamental-limitations literature (Havre & Skogestad give the general pole/zero theorem; the delay form is in Skogestad & Postlethwaite's textbook, which I did not retrieve to check the equation number). All four computed peaks respect the bound.

This is the delay bound in usable form: to hold the wormhole from distance d, the controller's model of the throat's response must be right, and must stay right, to better than exp(−lam × (2d − 0.5)). Each extra throat radius of distance costs a factor of 10.6.

## Full simulation (dx = 0.025, grid-balanced start)

| d | Observer-based | Predictor (estimator 3, gain 0.5) |
|---|---|---|
| 1 | open at 150; radius offset peaks at 0.18% and settles near 0.10% | open at 150; radius drifting, 0.67% and rising |
| 1.5 | open at 250; radius offset wanders between 0.1% and 0.77% | fails at 130 |
| 2 | fails at 99 | fails at 74 |

- At d = 1 the radius is now held, which the earlier designs never did.
- At d = 1.5 the radius offset rose to 0.77%, fell to 0.1% by t ≈ 190 and was back at 0.44% and rising at t = 250. That looks like a slow oscillation whose damping is unknown. It is not shown to be stable.
- At d = 2 the failure is slow and smooth (radius offset growing from t ≈ 25), not the fast sign-flipping oscillation of the predictor. The filter's state estimates become inconsistent with the readings, which is what exceeding a 0.65% margin looks like. The radius drift alone reaches that scale.
- A first attempt started the filter from zero and failed within 30 b at d ≥ 1.5 because of the start-up kick; the filter is now initialised from the first reading.

## Not done

- No fine-grid run of this design, and no run longer than 250 b.
- The design is not margin-optimal: it reaches a quarter to a half of the best possible margin. An H-infinity design would close that gap but cannot beat the bound.
- With w_radius reduced to 0.01, lifetimes were worse (d = 1.5: 90 b; d = 2: 74 b).
- Disturbance rejection (the "traveller" pulse) was not retested with this design.
