# Reproducibility and Evidence

This project treats experiment provenance as part of the implementation rather
than as an afterthought.

## Reference Software Environment

The audited RL environment used:

| Component | Version |
|---|---|
| ROS | ROS 2 Jazzy |
| Python | 3.12 |
| Gymnasium | 1.3.0 |
| Stable-Baselines3 | 2.9.0 |
| sb3-contrib | 2.9.0 |
| PyTorch | 2.9.1+cpu |

Formal run metadata independently records package versions and the Python
executable used at runtime.

## Classical Baseline

Frozen frontier implementation:

```text
40093e4
```

Formal baseline collection commit:

```text
76c7959
```

Five valid baseline repetitions are preserved in:

```text
experiments/baseline/runs/
```

The compact formal result table is:

```text
experiments/baseline/formal_results.csv
```

SHA-256 hashes of raw baseline evidence are preserved in:

```text
experiments/baseline/raw_run_sha256.txt
```

The formal procedure is documented in:

```text
docs/experiments/baseline_protocol.md
```

## Reference PPO Training Run

Formal run ID:

```text
formal_train64_seed0_004
```

Training source commit:

```text
1b9dec9312f22bb2185f5a9a95fe1bdf51bfc858
```

Configuration:

```text
algorithm       MaskablePPO
policy          MultiInputPolicy
timesteps       64
n_steps         16
batch_size      8
n_epochs        10
seed            0
device          cpu
```

Recorded outcome:

```text
physical transitions    64
physical episodes       10
optimization cycles      4
model _n_updates         40
navigation successes    63
navigation failures      1
```

Initial policy fingerprint:

```text
b3779dc5e9850bf2eee90ccb0e4c16c737e0e1e99e4db1595f9ce58767f62385
```

Final policy fingerprint:

```text
74f674467f629a43f8d9dbe4fb1f994def5e9adeddc5c01a22b97a6206bc3854
```

Initial checkpoint SHA-256:

```text
4ab545e8e44affa07bea16a1c7f998d1ad4e14353954a6807dc5d81a55b442d3
```

Final checkpoint SHA-256:

```text
d4f873f2896174a295a238ef61f7fd2f5610c4ee4b54ce02794146788b3d255f
```

The changed policy fingerprint establishes that PPO optimization changed model
parameters. Performance claims are based separately on frozen evaluation.

## Frozen Evaluation Contract

Evaluation source commit:

```text
e4b0e135a922de35df7f27d78a22a72a8425d2aa
```

The evaluator:

1. loads a serialized MaskablePPO checkpoint,
2. computes its policy fingerprint,
3. uses deterministic masked actions,
4. executes physical Gazebo/Nav2/SLAM episodes,
5. performs no call to `learn()`,
6. records zero optimization rows,
7. recomputes the policy fingerprint after evaluation,
8. fails if the fingerprint changed.

This separates training evidence from performance evidence.

## Repeated Trained-vs-Untrained Evaluation

Comparison ID:

```text
formal_eval_initial_vs_trained_5x_001
```

Design:

```text
5 frozen untrained-policy episodes
5 frozen trained-policy episodes
interleaved execution order
deterministic policy actions
learning disabled
```

All ten runs completed with zero optimization updates and unchanged policy
fingerprints.

The trained model showed, descriptively:

```text
mean return              +20.36%
mean path length         -17.99%
mean episode time        -19.89%
mean navigation actions  -28.57%
mean area per meter      +20.18%
mean area per second     +21.49%
```

Final explored area remained effectively unchanged.

This comparison evaluates whether the training changed policy behavior. It is
separate from the classical-baseline comparison.

## Trained RL vs Classical Frontier Baseline

Repository-facing results are preserved in:

```text
experiments/comparison/frontier_vs_trained_rl.csv
experiments/comparison/frontier_vs_trained_rl.json
docs/experiments/rl_vs_frontier_results.md
```

Across five repetitions per method:

```text
explored area      -0.014%
path length        -7.47%
area per meter     +6.45%
navigation goals  -52.83%
navigation success 100% for both
```

These are descriptive statistics.

## Decision-Level Evidence

A training step records:

- episode and global step indexes,
- UTC timestamp,
- physical step duration,
- episode elapsed time,
- selected action,
- exact pre-action mask,
- valid action count,
- goal coordinates,
- reward,
- cumulative return,
- map area gain,
- explored area,
- physical path delta,
- cumulative path,
- robot position,
- Nav2 acceptance/status/success,
- map revision,
- next action mask,
- termination state.

An episode record contains:

- reset timestamp,
- number of steps,
- episode return,
- initial/final explored area,
- initial/final path length,
- final elapsed time,
- navigation successes/failures,
- termination reason.

Each PPO optimization cycle records:

- optimization index,
- number of environment timesteps,
- PPO update count,
- learning rate,
- entropy loss,
- policy-gradient loss,
- value loss,
- approximate KL divergence,
- clipping fraction,
- explained variance,
- policy fingerprint.

## Experiment Provenance

Every formal run records:

```text
metadata.json
config.json
resolved_model.json
steps.csv
episodes.csv
updates.csv
summary.json
```

Training runs additionally preserve checkpoints and SHA-256 hashes.

Console output and the exact invocation are retained for sealed runs.

Formal evidence directories are immutable after collection.

## Evidence Integrity

Formal experiment packaging uses:

```text
evidence_manifest.sha256
```

The evidence directory can then be archived and the archive itself hashed.

This provides two integrity layers:

1. per-file SHA-256 verification,
2. archive-level SHA-256 verification.

## Known Limitations

### Simulator seed

Gymnasium reset seeds are recorded.

The current Gazebo launch command does not explicitly pin the simulator random
seed, so repeated trials are not claimed to be exact stochastic pairs.

### Different stopping rules

The frozen classical baseline uses an approximately 300-second formal horizon.

The RL evaluator may terminate earlier after confirmed frontier exhaustion.

Consequently, cross-method elapsed time is deliberately excluded from the
primary direct comparison.

### Sample count

Five physical repetitions per method are sufficient for descriptive statistics
but not for a strong statistical-significance claim.

### Simulation

The current study is conducted in TurtleBot3 simulation. A physical-robot study
remains future work.
