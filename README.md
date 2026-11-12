# wormhole-feedback

[![DOI](https://zenodo.org/badge/1454717806.svg)](https://doi.org/10.5281/zenodo.24150493)

Holding an unstable wormhole open with a feedback controller, in the full Einstein equations.

The Ellis–Bronnikov wormhole is an exact solution of general relativity, held open by a phantom (negative-energy) scalar field, and it is unstable: any perturbation grows as e^(λt) with λ ≈ 1.18 (time in light-crossing times of the throat). This repository contains:

- a validated solver for spherically symmetric Einstein–scalar evolution that reproduces the published instability rate (1.185 vs 1.182);
- a system-identified model of how the throat responds to pulses of phantom field;
- feedback controllers that hold the throat open (225 light-crossing times in the longest run), including an observer-based LQG design with a computed robustness margin;
- the delay bound: a controller at distance d from the throat must know the throat's response to better than e^(−λ(2d − 0.5)), about ten times tighter per throat radius;
- a browser version of the solver and controller that runs live: https://davidvillero.github.io/wormhole-feedback/.

Write-up: *[Medium article, link to add]*
Live simulation: https://davidvillero.github.io/wormhole-feedback/ (the same page is `index.html` and `wormhole-control-room.html` here; open either in a browser to run it offline)

This is not a faster-than-light result, and not peer reviewed. Limits: spherical symmetry only, ideal sensors and actuators, and the phantom field may not exist in nature. Corrections are very welcome.

## Layout

| Path | What |
|---|---|
| `wormhole_solver.py`, `derive_check.py`, `run_suite.py`, `analyze.py` | Single-field solver, symbolic check against the Einstein tensor, validation suite |
| `control_solver.py` | Two-field solver (phantom + ordinary scalar) with pulse injection that re-solves the constraints |
| `sysid.py`, `sysid_pair.py` | Pulse-response measurements (the pulse-pair trick that cancels the growing mode) |
| `control_regulator.py`, `control_feedback.py` | Predictor-based regulator |
| `control_lqg.py`, `control_lqg_run.py` | Kalman filter + LQR design and robustness margins |
| `rescheck.py`, `sweep.py` | Resolution check and controller-distance sweep |
| `*_report.py`, `results_*.json`, `*.png` | Reports, results and figures for each step |
| `runs/` | Raw run outputs (JSON, NumPy `.npz`, and Python pickles of long runs) |
| `live/` | JavaScript port: solver, controller, Web Worker, validation scripts |
| `article/` | Figures used in the write-up |
| `NOTEBOOK.md` | The full lab notebook: every step, check, failure and caveat, in order |

## Running

Python 3 with numpy, matplotlib and sympy.

    python3 derive_check.py      # symbolic verification, ~1 min
    python3 run_suite.py cmfx    # validation runs
    python3 analyze.py           # results.json, validation.png

The commands for each control step are in `NOTEBOOK.md`. The pickles in `runs/` are checkpoints of long runs; as with any pickle, only load them if you trust the source.

## How this was made

The work was done with AI coding agents (Anthropic's Claude for the solver, controllers and browser simulation), directed by the author. Every number in the write-up comes from code in this repository.

## References

- González, Guzmán & Sarbach, linear stability of the Ellis–Bronnikov wormhole: https://arxiv.org/abs/0806.0608
- González, Guzmán & Sarbach, nonlinear evolution: https://arxiv.org/abs/0806.1370
- Shinkai & Hayward, Fate of the first traversible wormhole (2002): https://arxiv.org/abs/gr-qc/0205041
- Xu, Chew & Yeom, Dynamics of the Bronnikov–Ellis wormhole (2025): https://arxiv.org/abs/2503.07610
- Havre & Skogestad, Fundamental limitations for control of unstable plants (1997)

## Citing

If you use this code or its results, please cite it: GitHub's "Cite this repository" button (from `CITATION.cff`) gives the reference. Archived on Zenodo: version v1.0.0 is [10.5281/zenodo.24150494](https://doi.org/10.5281/zenodo.24150494); [10.5281/zenodo.24150493](https://doi.org/10.5281/zenodo.24150493) always resolves to the latest version.

## License

MIT, see `LICENSE`.
