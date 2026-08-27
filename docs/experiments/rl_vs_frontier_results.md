# Trained RL vs Classical Frontier Exploration

## Experimental Status

This document reports a descriptive physical comparison between:

- the frozen classical nearest-frontier baseline, and
- the frozen MaskablePPO Active SLAM policy trained in
  `formal_train64_seed0_004`.

Both methods have five valid physical repetitions.

The RL checkpoint was evaluated with deterministic masked actions and
**no learning during evaluation**.

## Main Result

| Metric | Frontier baseline | Trained RL | Difference |
|---|---:|---:|---:|
| Newly explored area (m²) | 18.516501 ± 0.056668 | 18.514001 ± 0.040257 | -0.014% |
| Path length (m) | 15.262120 ± 2.340000 | 14.122536 ± 1.001662 | -7.47% |
| Area per meter (m²/m) | 1.236241 ± 0.188623 | 1.315937 ± 0.087889 | +6.45% |
| Navigation goals | 10.60 ± 1.14 | 5.00 ± 0.71 | -52.83% |
| Navigation success rate | 1.000 | 1.000 | +0.00% |

Values are mean ± sample standard deviation across five physical runs.

## Interpretation

The trained RL policy reached essentially the same final explored area as
the classical frontier baseline: the mean difference was only
-0.014%.

At that coverage level, the trained policy used approximately
7.5% less path length and achieved
approximately 6.4% higher explored
area per meter traveled.

It also issued approximately
52.8% fewer navigation goals while
maintaining a navigation success rate of
100.0%.

These results provide evidence that the learned frontier-selection policy
changed exploration behavior toward greater path efficiency.

## Important Evaluation Limitation

Elapsed time is intentionally **not** included in the direct
baseline-versus-RL table.

The classical baseline follows the frozen protocol with an approximately
300-second exploration horizon. The current RL evaluator terminates an
episode when the synchronized frontier detector confirms that no actionable
frontier remains, or when the episode horizon is reached.

Therefore the RL completion-time values and the baseline 300-second values
have different stopping rules and must not be presented as a direct timing
comparison.

## Reproducibility and Statistical Scope

The classical results come from the five frozen formal runs documented in
`experiments/baseline/formal_results.csv`.

The RL values come from the five frozen trained-checkpoint evaluations
preserved under the external evidence comparison
`formal_eval_initial_vs_trained_5x_001`.

No PPO optimization occurs during these evaluation runs, and policy
fingerprints are checked before and after evaluation.

The Gym reset seed is recorded for each RL evaluation. The current Gazebo
launch command does not explicitly control the simulator random seed, so the
trials are repeated physical evaluations rather than exact simulator-seed
pairs.

With five repetitions per method, the results are reported descriptively.
No strong statistical-significance claim is made from this sample alone.

## Headline Result

**The trained MaskablePPO frontier selector preserved essentially identical
map coverage while reducing mean path length by
7.5% and increasing explored area per
meter by 6.4% relative to the frozen
classical nearest-frontier baseline.**
