import csv
import hashlib
import json
import platform
import re
import subprocess
import sys

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


EVIDENCE_SCHEMA_VERSION = 1

RUN_KINDS = {
    'diagnostic',
    'formal',
}

PACKAGE_NAMES = (
    'sb3-contrib',
    'stable-baselines3',
    'gymnasium',
    'torch',
)

RUN_ID_PATTERN = re.compile(
    r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$'
)

STEP_COLUMNS = (
    'step_index',
    'episode_index',
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
)

EPISODE_COLUMNS = (
    'episode_index',
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
)


def utc_now_iso():
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


def _git_output(
    repo_root,
    *arguments,
):
    result = subprocess.run(
        (
            'git',
            *arguments,
        ),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.rstrip(
        '\n'
    )


def collect_runtime_provenance(
    repo_root,
):
    repo_root = Path(
        repo_root
    ).resolve()

    package_versions = {}

    for package_name in PACKAGE_NAMES:
        try:
            package_versions[
                package_name
            ] = version(
                package_name
            )

        except PackageNotFoundError:
            package_versions[
                package_name
            ] = None

    git_commit = _git_output(
        repo_root,
        'rev-parse',
        'HEAD',
    )

    git_branch = _git_output(
        repo_root,
        'branch',
        '--show-current',
    )

    git_status = _git_output(
        repo_root,
        'status',
        '--short',
    )

    return {
        'captured_at_utc': (
            utc_now_iso()
        ),
        'git_commit': git_commit,
        'git_branch': git_branch,
        'git_status_at_start': (
            git_status
        ),
        'git_worktree_clean': (
            git_status == ''
        ),
        'python_executable': (
            sys.executable
        ),
        'python_version': (
            platform.python_version()
        ),
        'platform': (
            platform.platform()
        ),
        'package_versions': (
            package_versions
        ),
    }


def sha256_file(path):
    path = Path(
        path
    )

    digest = hashlib.sha256()

    with path.open(
        'rb'
    ) as file_handle:
        while True:
            block = file_handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def _write_json_exclusive(
    path,
    payload,
):
    with Path(path).open(
        'x',
        encoding='utf-8',
    ) as file_handle:
        json.dump(
            payload,
            file_handle,
            indent=2,
            sort_keys=True,
        )

        file_handle.write(
            '\n'
        )


class ExperimentRecorder:
    def __init__(
        self,
        *,
        evidence_root,
        run_id,
        run_kind,
        config,
        repo_root=None,
        provenance=None,
    ):
        if not RUN_ID_PATTERN.fullmatch(
            run_id
        ):
            raise ValueError(
                'run_id contains unsupported characters.'
            )

        if run_kind not in RUN_KINDS:
            raise ValueError(
                'run_kind must be diagnostic or formal.'
            )

        if not isinstance(
            config,
            dict,
        ):
            raise TypeError(
                'config must be a dict.'
            )

        self.evidence_root = Path(
            evidence_root
        )

        self.run_id = run_id
        self.run_kind = run_kind

        self.run_dir = (
            self.evidence_root
            / 'runs'
            / run_id
        )

        self.run_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        if provenance is None:
            if repo_root is None:
                raise ValueError(
                    'repo_root is required when provenance '
                    'is not supplied.'
                )

            provenance = (
                collect_runtime_provenance(
                    repo_root
                )
            )

        metadata = {
            'evidence_schema_version': (
                EVIDENCE_SCHEMA_VERSION
            ),
            'run_id': run_id,
            'run_kind': run_kind,
            **provenance,
        }

        _write_json_exclusive(
            self.run_dir
            / 'metadata.json',
            metadata,
        )

        _write_json_exclusive(
            self.run_dir
            / 'config.json',
            config,
        )

        self.steps_path = (
            self.run_dir
            / 'steps.csv'
        )

        self.episodes_path = (
            self.run_dir
            / 'episodes.csv'
        )

        self._initialize_csv(
            self.steps_path,
            STEP_COLUMNS,
        )

        self._initialize_csv(
            self.episodes_path,
            EPISODE_COLUMNS,
        )

        self._step_count = 0
        self._episode_count = 0
        self._finished = False

    @staticmethod
    def _initialize_csv(
        path,
        columns,
    ):
        with Path(path).open(
            'x',
            encoding='utf-8',
            newline='',
        ) as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=columns,
            )

            writer.writeheader()

    @staticmethod
    def _append_csv(
        path,
        columns,
        payload,
    ):
        unknown = (
            set(payload)
            - set(columns)
        )

        if unknown:
            raise ValueError(
                'Unsupported CSV fields: '
                + ', '.join(
                    sorted(
                        unknown
                    )
                )
            )

        row = {
            column: payload.get(
                column,
                '',
            )
            for column in columns
        }

        with Path(path).open(
            'a',
            encoding='utf-8',
            newline='',
        ) as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=columns,
            )

            writer.writerow(
                row
            )

    def record_step(
        self,
        payload,
    ):
        if self._finished:
            raise RuntimeError(
                'Cannot record after finish().'
            )

        payload = dict(
            payload
        )

        expected_index = (
            self._step_count
        )

        supplied_index = payload.get(
            'step_index',
            expected_index,
        )

        if supplied_index != expected_index:
            raise ValueError(
                'step_index must be sequential.'
            )

        payload[
            'step_index'
        ] = expected_index

        self._append_csv(
            self.steps_path,
            STEP_COLUMNS,
            payload,
        )

        self._step_count += 1

    def record_episode(
        self,
        payload,
    ):
        if self._finished:
            raise RuntimeError(
                'Cannot record after finish().'
            )

        payload = dict(
            payload
        )

        expected_index = (
            self._episode_count
        )

        supplied_index = payload.get(
            'episode_index',
            expected_index,
        )

        if supplied_index != expected_index:
            raise ValueError(
                'episode_index must be sequential.'
            )

        payload[
            'episode_index'
        ] = expected_index

        self._append_csv(
            self.episodes_path,
            EPISODE_COLUMNS,
            payload,
        )

        self._episode_count += 1

    def record_checkpoint_hash(
        self,
        checkpoint_path,
        *,
        label='model',
    ):
        if self._finished:
            raise RuntimeError(
                'Cannot record after finish().'
            )

        if not RUN_ID_PATTERN.fullmatch(
            label
        ):
            raise ValueError(
                'checkpoint label contains '
                'unsupported characters.'
            )

        checkpoint_path = Path(
            checkpoint_path
        )

        digest = sha256_file(
            checkpoint_path
        )

        hash_path = (
            self.run_dir
            / f'{label}.sha256'
        )

        with hash_path.open(
            'x',
            encoding='utf-8',
        ) as file_handle:
            file_handle.write(
                f'{digest}  '
                f'{checkpoint_path.name}\n'
            )

        return digest

    def finish(
        self,
        summary,
    ):
        if self._finished:
            raise RuntimeError(
                'finish() may only be called once.'
            )

        payload = {
            **dict(
                summary
            ),
            'finished_at_utc': (
                utc_now_iso()
            ),
            'recorded_steps': (
                self._step_count
            ),
            'recorded_episodes': (
                self._episode_count
            ),
        }

        _write_json_exclusive(
            self.run_dir
            / 'summary.json',
            payload,
        )

        self._finished = True

        return payload
