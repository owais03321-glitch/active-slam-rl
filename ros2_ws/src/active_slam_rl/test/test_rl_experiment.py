import csv
import hashlib
import json

import pytest

from active_slam_rl.rl_experiment import (
    ExperimentRecorder,
    sha256_file,
)


def fixed_provenance():
    return {
        'captured_at_utc': (
            '2026-08-27T00:00:00+00:00'
        ),
        'git_commit': 'abc123',
        'git_branch': 'rl-active-slam',
        'git_status_at_start': '',
        'git_worktree_clean': True,
        'python_executable': (
            '/project/.venv/bin/python3'
        ),
        'python_version': '3.12.3',
        'platform': 'test-platform',
        'package_versions': {
            'sb3-contrib': '2.9.0',
            'stable-baselines3': '2.9.0',
            'gymnasium': '1.3.0',
            'torch': '2.9.1+cpu',
        },
    }


def make_recorder(
    tmp_path,
    *,
    run_id='diag_001',
):
    return ExperimentRecorder(
        evidence_root=tmp_path,
        run_id=run_id,
        run_kind='diagnostic',
        config={
            'seed': 0,
            'n_steps': 2,
            'batch_size': 2,
        },
        provenance=fixed_provenance(),
    )


def test_recorder_writes_immutable_manifest(
    tmp_path,
):
    recorder = make_recorder(
        tmp_path
    )

    metadata = json.loads(
        (
            recorder.run_dir
            / 'metadata.json'
        ).read_text()
    )

    config = json.loads(
        (
            recorder.run_dir
            / 'config.json'
        ).read_text()
    )

    assert metadata[
        'run_id'
    ] == 'diag_001'

    assert metadata[
        'run_kind'
    ] == 'diagnostic'

    assert metadata[
        'git_commit'
    ] == 'abc123'

    assert metadata[
        'git_worktree_clean'
    ] is True

    assert config == {
        'batch_size': 2,
        'n_steps': 2,
        'seed': 0,
    }

    with pytest.raises(
        FileExistsError
    ):
        make_recorder(
            tmp_path
        )


def test_invalid_run_identity_is_rejected(
    tmp_path,
):
    with pytest.raises(
        ValueError
    ):
        ExperimentRecorder(
            evidence_root=tmp_path,
            run_id='../bad',
            run_kind='diagnostic',
            config={},
            provenance=fixed_provenance(),
        )

    with pytest.raises(
        ValueError
    ):
        ExperimentRecorder(
            evidence_root=tmp_path,
            run_id='run_001',
            run_kind='unclassified',
            config={},
            provenance=fixed_provenance(),
        )


def test_step_and_episode_rows_are_recorded(
    tmp_path,
):
    recorder = make_recorder(
        tmp_path
    )

    recorder.record_step(
        {
            'episode_index': 0,
            'action': 3,
            'valid_action_count': 7,
            'goal_x': 1.5,
            'goal_y': -0.25,
            'reward': 2.75,
            'cumulative_episode_return': 2.75,
            'area_gain_m2': 2.9,
            'path_delta_m': 1.5,
            'navigation_accepted': True,
            'navigation_status': 4,
            'navigation_succeeded': True,
            'map_revision': 5,
            'terminated': False,
            'truncated': False,
        }
    )

    recorder.record_episode(
        {
            'steps': 1,
            'episode_return': 2.75,
            'terminated': False,
            'truncated': True,
            'outcome': 'smoke_limit',
        }
    )

    with recorder.steps_path.open(
        newline='',
    ) as file_handle:
        rows = list(
            csv.DictReader(
                file_handle
            )
        )

    assert len(rows) == 1
    assert rows[0][
        'step_index'
    ] == '0'
    assert rows[0][
        'action'
    ] == '3'
    assert rows[0][
        'navigation_succeeded'
    ] == 'True'

    with recorder.episodes_path.open(
        newline='',
    ) as file_handle:
        episodes = list(
            csv.DictReader(
                file_handle
            )
        )

    assert len(episodes) == 1
    assert episodes[0][
        'episode_index'
    ] == '0'
    assert episodes[0][
        'outcome'
    ] == 'smoke_limit'


def test_step_indices_must_be_sequential(
    tmp_path,
):
    recorder = make_recorder(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match='sequential',
    ):
        recorder.record_step(
            {
                'step_index': 4,
            }
        )


def test_finish_writes_counts_once(
    tmp_path,
):
    recorder = make_recorder(
        tmp_path
    )

    recorder.record_step(
        {
            'episode_index': 0,
            'action': 1,
        }
    )

    recorder.record_episode(
        {
            'steps': 1,
            'episode_return': 1.0,
            'outcome': 'diagnostic',
        }
    )

    summary = recorder.finish(
        {
            'status': 'complete',
            'total_reward': 1.0,
        }
    )

    assert summary[
        'recorded_steps'
    ] == 1

    assert summary[
        'recorded_episodes'
    ] == 1

    persisted = json.loads(
        (
            recorder.run_dir
            / 'summary.json'
        ).read_text()
    )

    assert persisted[
        'status'
    ] == 'complete'

    assert persisted[
        'recorded_steps'
    ] == 1

    with pytest.raises(
        RuntimeError
    ):
        recorder.finish(
            {}
        )

    with pytest.raises(
        RuntimeError
    ):
        recorder.record_step(
            {}
        )


def test_checkpoint_hash_is_auditable(
    tmp_path,
):
    recorder = make_recorder(
        tmp_path
    )

    checkpoint = (
        tmp_path
        / 'model.zip'
    )

    checkpoint.write_bytes(
        b'known-model-bytes'
    )

    expected = hashlib.sha256(
        b'known-model-bytes'
    ).hexdigest()

    assert sha256_file(
        checkpoint
    ) == expected

    digest = (
        recorder
        .record_checkpoint_hash(
            checkpoint
        )
    )

    assert digest == expected

    hash_text = (
        recorder.run_dir
        / 'model.sha256'
    ).read_text()

    assert hash_text == (
        f'{expected}  model.zip\n'
    )


def test_unknown_csv_fields_fail_closed(
    tmp_path,
):
    recorder = make_recorder(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match='Unsupported CSV fields',
    ):
        recorder.record_step(
            {
                'invented_metric': 123,
            }
        )


def test_evidence_schema_contains_decision_audit_fields():
    from active_slam_rl.rl_experiment import (
        EPISODE_COLUMNS,
        STEP_COLUMNS,
    )

    required_step_fields = {
        'step_started_at_utc',
        'step_duration_s',
        'episode_elapsed_s',
        'action',
        'action_mask_bits',
        'valid_action_count',
        'goal_x',
        'goal_y',
        'reward',
        'cumulative_episode_return',
        'area_gain_m2',
        'explored_area_m2',
        'path_delta_m',
        'cumulative_path_m',
        'robot_x',
        'robot_y',
        'navigation_accepted',
        'navigation_status',
        'navigation_succeeded',
        'map_revision',
        'next_action_mask_bits',
        'next_valid_action_count',
        'terminated',
        'truncated',
    }

    required_episode_fields = {
        'episode_reset_at_utc',
        'steps',
        'episode_return',
        'initial_explored_area_m2',
        'final_explored_area_m2',
        'initial_path_m',
        'final_path_m',
        'final_episode_elapsed_s',
        'navigation_successes',
        'navigation_failures',
        'terminated',
        'truncated',
        'outcome',
    }

    assert required_step_fields.issubset(
        STEP_COLUMNS
    )

    assert required_episode_fields.issubset(
        EPISODE_COLUMNS
    )
