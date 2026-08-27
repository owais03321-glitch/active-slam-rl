# Varied-Start Frozen-Policy Robustness Check

This experiment is a targeted sanity check against memorization of one exact
initial robot pose or one exact trajectory.

The already-trained MaskablePPO checkpoint was evaluated with learning
disabled from three altered TurtleBot3 initial poses in the same simulation
world. No retraining or fine-tuning occurred.

## Results

| Run | Initial pose (x, y, yaw) | Steps | Area gain | Path | Area/m | Nav success |
|---|---:|---:|---:|---:|---:|---:|
| pose_robustness_p1_001 | (-2.00, -0.50, 1.5708) | 5 | 10.613 m² | 12.497 m | 0.849 m²/m | 5/5 |
| pose_robustness_p2_001 | (-1.75, -0.50, -1.5708) | 9 | 17.538 m² | 15.814 m | 1.109 m²/m | 9/9 |
| pose_robustness_p3_001 | (-2.00, -0.25, 3.1416) | 7 | 14.803 m² | 18.831 m | 0.786 m²/m | 7/7 |

Aggregate:

- 3/3 episodes completed.
- 21/21 navigation goals succeeded (100%).
- Mean newly explored area: 14.318 m².
- Mean path length: 15.714 m.
- Mean area per meter: 0.915 m²/m.
- Recorded PPO optimization updates: 0 in every run.
- The policy fingerprint was unchanged before and after every run.

## Interpretation

The result is evidence against severe memorization of one exact start pose
or one exact trajectory: the same frozen policy remained operational and
completed exploration from three different initial configurations.

It is not evidence of broad environment generalization. All three runs use
the same simulation world, so unseen-world evaluation remains future work.

The runs are descriptive robustness checks rather than a statistical study.

## Evidence

- `experiments/rl/varied_start_robustness_3x.csv`
- `experiments/rl/varied_start_robustness_3x.json`

The raw decision-level evidence remains in the external experiment evidence
store under the three run IDs shown above.
