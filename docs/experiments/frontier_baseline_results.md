# Frontier Baseline Formal Results

## Experimental Status

Five formal frontier-baseline repetitions were collected successfully.

- Collection Git commit: `76c7959`
- Frozen frontier implementation commit: `40093e4`
- World: `nav2_tb3_default_simulation`
- Formal horizon: `exploration_elapsed_s <= 300.0 s`
- Valid formal runs: 5 / 5

All five runs started with a clean Git working tree and a fresh
Gazebo/Nav2/SLAM process. No metrics or explorer traceback, RCLError,
exception, or fatal error was observed during validation.

Coverage percentage, T90, and T99 remain disabled because
`target_area_m2` has not yet been established.

## Individual Formal Runs

| Run | Time (s) | New area (m²) | Path (m) | Area/path (m²/m) | Goals | Success rate |
|---|---:|---:|---:|---:|---:|---:|
| frontier_baseline_01 | 299.841 | 18.425001 | 12.433792 | 1.481849 | 11 | 1.000 |
| frontier_baseline_02 | 299.874 | 18.562501 | 15.795200 | 1.175199 | 11 | 1.000 |
| frontier_baseline_03 | 299.862 | 18.565001 | 18.343073 | 1.012099 | 12 | 1.000 |
| frontier_baseline_04 | 299.916 | 18.515001 | 13.487630 | 1.372739 | 10 | 1.000 |
| frontier_baseline_05 | 299.934 | 18.515001 | 16.250906 | 1.139321 | 9 | 1.000 |

## Aggregate Statistics

| Metric | Mean | Std. dev. | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Newly explored area (m²) | 18.516501 | 0.056668 | 18.425001 | 18.565001 |
| Path length (m) | 15.262120 | 2.340000 | 12.433792 | 18.343073 |
| Area per meter (m²/m) | 1.236241 | 0.188623 | 1.012099 | 1.481849 |
| Goal success rate | 1.000000 | 0.000000 | 1.000000 | 1.000000 |

The reported standard deviation is the sample standard deviation across
the five independent repetitions.

## Navigation Outcomes

- Goals observed: 53
- Goals succeeded: 53
- Goals aborted: 0
- Goals canceled: 0
- Aggregate observed goal outcome: 53/53 successful

## Interpretation

Newly explored area was highly repeatable across the five repetitions,
while path length showed substantially greater run-to-run variation.
Consequently, exploration efficiency varied more than final explored area.

These results define the frozen classical frontier baseline against which
the later reinforcement-learning Active SLAM policy will be evaluated
using the same formal horizon and metric definitions.

Raw run files are preserved separately and their SHA-256 hashes are recorded
in `experiments/baseline/raw_run_sha256.txt`.
