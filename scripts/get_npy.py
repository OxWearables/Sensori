#!/usr/bin/env python3
"""Convert one wearable recording into model-ready daily NumPy arrays.

This file is intentionally standalone: copy it, install its dependencies, and
run it directly. It accepts ActiPy-supported CWA, GT3X, and BIN recordings
(plain or gzip-compressed), plus CSV/CSV.GZ files containing ``time,x,y,z``.

ActiPy's device readers also require Java 8 or newer. Example::

    python get_npy.py --file participant.cwa.gz --output processed

The paper preprocessing is fixed: gravity calibration, a 5 Hz low-pass filter,
10 Hz resampling, non-wear detection, at least 22 hours of wear per complete
calendar day, mean ENMO <= 200 mg, and fewer than 10 recording interrupts.

Eligible days are written chronologically as ``day_0.npy``, ``day_1.npy``, ...
Each file contains acceleration in g, ordered XYZ, with dtype ``float32`` and
shape ``(2880, 300, 3)`` (30-second windows at 10 Hz). ``info.json`` and
``wear_duration.csv`` record provenance and quality-control decisions. The
processed waveform is not retained. Exit status 3 means processing succeeded
but strict quality control found no eligible output day.

Changing the scientific constants in this file makes its arrays incompatible
with the released paper model.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger('colap.get_npy')

SCRIPT_VERSION = '1.0.0'
PARSER_SCHEMA_VERSION = 3
REQUIRED_ACTIPY_VERSION = '3.4.0'

SECONDS_PER_DAY = 24 * 60 * 60
TARGET_SAMPLE_RATE_HZ = 10
LOWPASS_HZ = 5.0
MIN_WEAR_HOURS = 22.0
WINDOW_SECONDS = 30
MAX_MEAN_ACCELERATION_MG = 200.0
MAX_INTERRUPTS_EXCLUSIVE = 10

DEVICE_SUFFIXES = ('.cwa', '.cwa.gz', '.gt3x', '.gt3x.gz', '.bin', '.bin.gz')
CSV_SUFFIXES = ('.csv', '.csv.gz')
SUPPORTED_SUFFIXES = DEVICE_SUFFIXES + CSV_SUFFIXES

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INELIGIBLE = 3


@dataclass(frozen=True)
class _ProcessingConfig:
    """Scientific settings; production callers use the fixed defaults."""

    target_sample_rate_hz: int = TARGET_SAMPLE_RATE_HZ
    lowpass_hz: float = LOWPASS_HZ
    min_wear_hours: float = MIN_WEAR_HOURS
    window_seconds: int = WINDOW_SECONDS
    max_mean_acceleration_mg: float = MAX_MEAN_ACCELERATION_MG
    max_interrupts_exclusive: int = MAX_INTERRUPTS_EXCLUSIVE
    seconds_per_day: int = SECONDS_PER_DAY
    input_sample_rate_hz: float | None = None

    @property
    def samples_per_day(self) -> int:
        return self.target_sample_rate_hz * self.seconds_per_day

    @property
    def samples_per_window(self) -> int:
        return self.target_sample_rate_hz * self.window_seconds

    def validate(self) -> None:
        if self.target_sample_rate_hz <= 0:
            raise ValueError('target sample rate must be greater than zero')
        if self.lowpass_hz <= 0:
            raise ValueError('low-pass cutoff must be greater than zero')
        if not 0 <= self.min_wear_hours <= 24:
            raise ValueError('minimum wear duration must be between 0 and 24 hours')
        if self.window_seconds <= 0 or self.seconds_per_day % self.window_seconds:
            raise ValueError('window duration must divide evenly into a complete day')
        if self.max_mean_acceleration_mg <= 0:
            raise ValueError('maximum mean acceleration must be greater than zero')
        if self.max_interrupts_exclusive <= 0:
            raise ValueError('interrupt threshold must be greater than zero')
        if self.input_sample_rate_hz is not None and (
            not np.isfinite(self.input_sample_rate_hz) or self.input_sample_rate_hz <= 0
        ):
            raise ValueError('input sample rate must be finite and greater than zero')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _OutputPaths:
    recording_dir: Path
    info: Path
    wear_duration: Path


@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    mean_acceleration_mg: float | None
    criteria: dict[str, bool]


@dataclass(frozen=True)
class RunResult:
    recording_dir: Path
    eligible: bool
    days_written: int
    skipped_existing: bool = False


class _NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Convert one wearable recording into inference-ready daily NPY files.',
        epilog=(
            'Outputs are fixed to the paper representation: float32 XYZ in g, '
            'shape (2880, 300, 3). Exit 3 means that strict QC found no eligible day.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-f',
        '--file',
        required=True,
        type=Path,
        help='input .cwa[.gz], .gt3x[.gz], .bin[.gz], or time,x,y,z CSV[.gz]',
    )
    parser.add_argument('-o', '--output', required=True, type=Path, help='root output directory')
    parser.add_argument(
        '--output-name',
        help='safe output subdirectory name (default: complete filename without its recognized extension)',
    )
    parser.add_argument(
        '--input-sample-rate',
        type=float,
        default=None,
        metavar='HZ',
        help='nominal source rate for CSV input; inferred from timestamps when omitted',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='replace only info.json, wear_duration.csv, and day_*.npy in this recording directory',
    )
    parser.add_argument('--quiet-actipy', action='store_true', help='suppress ActiPy progress messages')
    parser.add_argument(
        '--log-level',
        choices=('DEBUG', 'INFO', 'WARNING', 'ERROR'),
        default='INFO',
        help='logging verbosity',
    )
    return parser


def strip_supported_suffix(filename: str) -> str:
    lower_name = filename.lower()
    for suffix in sorted(SUPPORTED_SUFFIXES, key=len, reverse=True):
        if lower_name.endswith(suffix):
            return filename[: -len(suffix)]
    supported = ', '.join(SUPPORTED_SUFFIXES)
    raise ValueError(f'unsupported input extension for {filename!r}; supported suffixes: {supported}')


def derive_output_name(input_path: Path, requested_name: str | None) -> str:
    output_name = requested_name if requested_name is not None else strip_supported_suffix(input_path.name)
    output_name = str(output_name).strip()
    if not output_name or output_name in {'.', '..'} or '/' in output_name or '\\' in output_name:
        raise ValueError('output name must be a non-empty directory name without path separators')
    return output_name


def is_csv_input(path: Path) -> bool:
    return path.name.lower().endswith(CSV_SUFFIXES)


def load_csv_waveform(path: Path) -> pd.DataFrame:
    """Read and validate a custom waveform CSV without silently dropping rows."""

    required = {'time', 'x', 'y', 'z'}
    data = pd.read_csv(path, usecols=lambda column: column in required)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f'CSV is missing required columns: {", ".join(missing)}')

    try:
        data['time'] = pd.to_datetime(data['time'], errors='raise')
        for column in ('x', 'y', 'z'):
            data[column] = pd.to_numeric(data[column], errors='raise')
    except (TypeError, ValueError) as exc:
        raise ValueError(f'CSV contains invalid time or acceleration values: {exc}') from exc

    if len(data) < 2:
        raise ValueError('CSV must contain at least two samples')
    if data['time'].isna().any() or data[['x', 'y', 'z']].isna().any().any():
        raise ValueError('input CSV must not contain missing time or acceleration values')
    if not np.isfinite(data[['x', 'y', 'z']].to_numpy()).all():
        raise ValueError('input CSV must contain only finite acceleration values')
    if data['time'].duplicated().any() or not data['time'].is_monotonic_increasing:
        raise ValueError('input CSV timestamps must be unique and strictly increasing')

    return data.loc[:, ['time', 'x', 'y', 'z']].set_index('time')


def infer_sample_rate(index: pd.DatetimeIndex) -> float:
    differences = index.to_series().diff().dt.total_seconds().dropna()
    differences = differences[differences > 0]
    if differences.empty:
        raise ValueError('cannot infer CSV sample rate from the timestamps')
    median_interval = float(differences.median())
    if not np.isfinite(median_interval) or median_interval <= 0:
        raise ValueError('cannot infer a valid CSV sample rate from the timestamps')
    sample_rate = 1.0 / median_interval
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError('cannot infer a valid CSV sample rate from the timestamps')
    return sample_rate


def _require_actipy() -> Any:
    try:
        import actipy
    except ImportError as exc:
        raise RuntimeError(
            'ActiPy is required. Install dependencies with '
            f'`pip install actipy=={REQUIRED_ACTIPY_VERSION} numpy pandas` and ensure Java 8+ is available.'
        ) from exc

    try:
        installed_version = importlib.metadata.version('actipy')
    except importlib.metadata.PackageNotFoundError:
        installed_version = getattr(actipy, '__version__', None)
    if installed_version != REQUIRED_ACTIPY_VERSION:
        raise RuntimeError(
            f'ActiPy {REQUIRED_ACTIPY_VERSION} is required for paper-compatible preprocessing; '
            f'found {installed_version or "an unknown version"}. Install it with '
            f'`pip install actipy=={REQUIRED_ACTIPY_VERSION}`.'
        )
    return actipy


def _normalise_actipy_frame(data: pd.DataFrame, label: str) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f'{label} data must be a pandas DataFrame')
    if isinstance(data.index, pd.DatetimeIndex):
        data = data.reset_index()
    elif 'time' not in data.columns:
        raise ValueError(f'{label} data must use a DatetimeIndex or contain a time column')
    if 'time' not in data.columns and 'index' in data.columns:
        data = data.rename(columns={'index': 'time'})

    missing = sorted({'time', 'x', 'y', 'z'}.difference(data.columns))
    if missing:
        raise ValueError(f'{label} data is missing columns: {", ".join(missing)}')
    result = data.loc[:, ['time', 'x', 'y', 'z']].copy()
    result['time'] = pd.to_datetime(result['time'], errors='raise')
    if result['time'].isna().any():
        raise ValueError(f'{label} data contains missing timestamps')
    if result['time'].duplicated().any() or not result['time'].is_monotonic_increasing:
        raise ValueError(f'{label} timestamps must be unique and strictly increasing')
    for column in ('x', 'y', 'z'):
        result[column] = pd.to_numeric(result[column], errors='raise').astype('float32')
    return result


def _metadata_bool(info: dict[str, Any], key: str) -> bool:
    value = info.get(key)
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes'}
    return bool(value)


def _validate_actipy_metadata(info: dict[str, Any], *, csv_input: bool) -> dict[str, Any]:
    if not isinstance(info, dict):
        raise TypeError('ActiPy metadata must be a dictionary')
    result = dict(info)
    if csv_input:
        result.setdefault('ReadOK', 1)
    required = {'ReadOK', 'CalibOK', 'LowpassOK', 'NumInterrupts'}
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f'ActiPy metadata is missing required fields: {", ".join(missing)}')
    try:
        interrupts = int(result['NumInterrupts'])
    except (TypeError, ValueError) as exc:
        raise ValueError('ActiPy NumInterrupts metadata must be an integer') from exc
    if interrupts < 0:
        raise ValueError('ActiPy NumInterrupts metadata must not be negative')
    result['NumInterrupts'] = interrupts
    return result


def _csv_interrupt_count(index: pd.DatetimeIndex, nominal_rate_hz: float) -> int:
    differences = index.to_series().diff().dt.total_seconds().dropna()
    expected_interval = 1.0 / nominal_rate_hz
    return int((differences > max(1.0, expected_interval * 1.5)).sum())


def read_and_process_recording(
    input_path: Path,
    config: _ProcessingConfig,
    *,
    actipy_module: Any | None = None,
    actipy_verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], float | None]:
    """Read the recording once, then derive the wear mask and the retained signal."""

    actipy_module = actipy_module or _require_actipy()
    processing = actipy_module.processing
    common_kwargs = {
        'lowpass_hz': config.lowpass_hz,
        'calibrate_gravity': True,
        'detect_nonwear': False,
        'resample_hz': None,
        'verbose': actipy_verbose,
    }
    csv_input = is_csv_input(input_path)
    inferred_rate: float | None = None

    if csv_input:
        source_data = load_csv_waveform(input_path)
        inferred_rate = infer_sample_rate(source_data.index)
        source_rate = config.input_sample_rate_hz or inferred_rate
        if source_rate <= 2 * config.lowpass_hz:
            raise ValueError(
                f'CSV input sample rate must be greater than {2 * config.lowpass_hz:g} Hz '
                f'to apply the {config.lowpass_hz:g} Hz low-pass filter; received {source_rate:g} Hz'
            )
        if config.input_sample_rate_hz is None:
            LOGGER.info('Inferred CSV input sample rate: %.6g Hz', source_rate)
        elif not np.isclose(source_rate, inferred_rate, rtol=0.01):
            LOGGER.warning(
                'Configured CSV input rate %.6g Hz differs from timestamp-derived rate %.6g Hz',
                source_rate,
                inferred_rate,
            )
        csv_interrupts = _csv_interrupt_count(source_data.index, source_rate)
        data, signal_info = actipy_module.process(source_data, source_rate, **common_kwargs)
        del source_data
        signal_info = dict(signal_info)
    else:
        data, signal_info = actipy_module.read_device(str(input_path), **common_kwargs)
        signal_info = dict(signal_info)

    if len(data) == 0:
        raise ValueError(f'ActiPy returned no samples for {input_path}')

    # flag_nonwear overwrites non-wear samples with NaN, so the mask and the retained
    # waveform need separate frames. Split here, in ActiPy's own mask-then-resample
    # order, rather than reading and calibrating the recording a second time.
    wear_data, wear_info = processing.flag_nonwear(data)
    signal_data, resample_info = processing.resample(data, config.target_sample_rate_hz)
    del data
    signal_info.update(resample_info)
    if csv_input:
        signal_info.setdefault('Filename', str(input_path))
        signal_info.setdefault('SampleRate', source_rate)
        signal_info.setdefault('ReadOK', 1)
        signal_info.setdefault('NumInterrupts', csv_interrupts)
    wear_data, _ = processing.resample(wear_data, config.target_sample_rate_hz)
    if not wear_data.index.equals(signal_data.index):
        raise ValueError('ActiPy processing passes returned different timestamp grids')
    wear_flags = wear_data[['x', 'y', 'z']].notna().all(axis=1).to_numpy(dtype=bool)
    del wear_data

    signal_frame = _normalise_actipy_frame(signal_data, 'signal processed')
    signal_frame['is_wear'] = wear_flags

    validated_info = _validate_actipy_metadata(dict(signal_info), csv_input=csv_input)
    return signal_frame, validated_info, dict(wear_info), inferred_rate


def _day_skip_reason(wear_ok: bool, complete_ok: bool) -> str:
    reasons: list[str] = []
    if not wear_ok:
        reasons.append('insufficient_wear')
    if not complete_ok:
        reasons.append('incomplete_or_missing_samples')
    return ';'.join(reasons)


def build_daily_summary(data: pd.DataFrame, config: _ProcessingConfig) -> pd.DataFrame:
    """Create one auditable QC row for every calendar day in the recording."""

    if data.empty:
        return pd.DataFrame(
            columns=(
                'time',
                'sample_count',
                'wear_samples',
                'wear_seconds',
                'wear_duration_ok',
                'complete_day',
                'day_ok',
                'output_file',
                'skip_reason',
            )
        )

    working = data.loc[:, ['time', 'x', 'y', 'z', 'is_wear']].copy()
    working['date'] = working['time'].dt.normalize()
    grouped = working.groupby('date', sort=True)
    summary = grouped.agg(sample_count=('is_wear', 'size'), wear_samples=('is_wear', 'sum'))
    finite_xyz = grouped[['x', 'y', 'z']].apply(lambda frame: bool(np.isfinite(frame.to_numpy()).all()))
    summary['wear_seconds'] = summary['wear_samples'] / config.target_sample_rate_hz
    summary['wear_duration_ok'] = summary['wear_seconds'] >= config.min_wear_hours * 60 * 60
    summary['complete_day'] = (summary['sample_count'] == config.samples_per_day) & finite_xyz
    summary['day_ok'] = summary['wear_duration_ok'] & summary['complete_day']
    summary['output_file'] = ''
    summary['skip_reason'] = [
        _day_skip_reason(bool(wear_ok), bool(complete_ok))
        for wear_ok, complete_ok in zip(summary['wear_duration_ok'], summary['complete_day'])
    ]
    return summary.reset_index().rename(columns={'date': 'time'})


def check_eligibility(
    data: pd.DataFrame,
    info: dict[str, Any],
    daily_summary: pd.DataFrame,
    config: _ProcessingConfig,
) -> EligibilityResult:
    magnitude = np.sqrt(
        data['x'].astype('float64') ** 2 + data['y'].astype('float64') ** 2 + data['z'].astype('float64') ** 2
    )
    computed_mean = float(((magnitude - 1.0).clip(lower=0) * 1000).mean())
    mean_acceleration_mg = computed_mean if np.isfinite(computed_mean) else None
    criteria = {
        'data_nonempty': bool(len(data)),
        'read_ok': _metadata_bool(info, 'ReadOK'),
        'calibration_ok': _metadata_bool(info, 'CalibOK'),
        'lowpass_ok': _metadata_bool(info, 'LowpassOK'),
        'interrupts_ok': int(info['NumInterrupts']) < config.max_interrupts_exclusive,
        'mean_acceleration_ok': bool(
            mean_acceleration_mg is not None and mean_acceleration_mg <= config.max_mean_acceleration_mg
        ),
        'eligible_day_available': bool(not daily_summary.empty and daily_summary['day_ok'].any()),
    }
    return EligibilityResult(all(criteria.values()), mean_acceleration_mg, criteria)


def reshape_day(day_data: pd.DataFrame, config: _ProcessingConfig) -> np.ndarray:
    xyz = day_data.loc[:, ['x', 'y', 'z']].to_numpy(dtype=np.float32, copy=True)
    expected_shape = (config.samples_per_day, 3)
    if xyz.shape != expected_shape:
        raise ValueError(f'expected day array {expected_shape}, received {xyz.shape}')
    if not np.isfinite(xyz).all():
        raise ValueError('day array contains NaN or infinite acceleration values')
    return xyz.reshape(-1, config.samples_per_window, 3)


def _source_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        'path': str(path.resolve()),
        'size_bytes': stat.st_size,
        'modified_time_ns': stat.st_mtime_ns,
    }


def _output_paths(output_root: Path, output_name: str) -> _OutputPaths:
    recording_dir = output_root / output_name
    return _OutputPaths(
        recording_dir=recording_dir,
        info=recording_dir / 'info.json',
        wear_duration=recording_dir / 'wear_duration.csv',
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if value is pd.NaT:
        return None
    return value


def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            'w', encoding='utf-8', prefix='.colap-', suffix='.tmp', dir=path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                _json_safe(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
                cls=_NumpyJSONEncoder,
            )
            handle.write('\n')
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_csv_dump(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            'w', encoding='utf-8', prefix='.colap-', suffix='.tmp', dir=path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        data.to_csv(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_npy_dump(data: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            'wb', prefix='.colap-', suffix='.tmp', dir=path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            np.save(handle, data, allow_pickle=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _generated_outputs(paths: _OutputPaths) -> list[Path]:
    outputs = [path for path in (paths.info, paths.wear_duration) if path.exists()]
    if paths.recording_dir.exists():
        outputs.extend(paths.recording_dir.glob('day_*.npy'))
    return sorted(set(outputs))


def _remove_generated_outputs(paths: _OutputPaths) -> None:
    for path in _generated_outputs(paths):
        path.unlink()


def _existing_completed_result(
    paths: _OutputPaths,
    source: dict[str, Any],
    config: _ProcessingConfig,
) -> RunResult | None:
    if not paths.info.is_file() or not paths.wear_duration.is_file():
        return None
    try:
        with paths.info.open(encoding='utf-8') as handle:
            info = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    parser_info = info.get('colap_parser', {})
    if parser_info.get('schema_version') != PARSER_SCHEMA_VERSION:
        return None
    if parser_info.get('source') != source or parser_info.get('parameters') != config.to_dict():
        return None
    status = parser_info.get('status')
    if status not in {'complete', 'ineligible'}:
        return None
    try:
        days_written = int(parser_info['days_written'])
    except (KeyError, TypeError, ValueError):
        return None
    day_files = list(paths.recording_dir.glob('day_*.npy'))
    eligible = bool(info.get('participant_eligibility', 0))
    if len(day_files) != days_written:
        return None
    if status == 'complete' and (not eligible or days_written < 1):
        return None
    if status == 'ineligible' and (eligible or days_written != 0):
        return None
    return RunResult(paths.recording_dir, eligible, days_written, skipped_existing=True)


def parse_recording(
    input_path: Path,
    output_root: Path,
    *,
    output_name: str | None = None,
    input_sample_rate_hz: float | None = None,
    overwrite: bool = False,
    actipy_module: Any | None = None,
    actipy_verbose: bool = True,
    _config: _ProcessingConfig | None = None,
) -> RunResult:
    """Process one recording; ``_config`` exists only to keep unit tests small."""

    config = _config or _ProcessingConfig(input_sample_rate_hz=input_sample_rate_hz)
    if _config is not None and input_sample_rate_hz is not None:
        raise ValueError('input_sample_rate_hz and _config cannot both be supplied')
    config.validate()
    input_path = input_path.expanduser()
    output_root = output_root.expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f'input recording not found: {input_path}')
    strip_supported_suffix(input_path.name)
    if config.input_sample_rate_hz is not None and not is_csv_input(input_path):
        raise ValueError('--input-sample-rate applies only to CSV/CSV.GZ input')

    output_name = derive_output_name(input_path, output_name)
    paths = _output_paths(output_root, output_name)
    source = _source_metadata(input_path)
    existing = _generated_outputs(paths)
    if existing and not overwrite:
        completed = _existing_completed_result(paths, source, config)
        if completed is not None:
            LOGGER.info('Matching completed output already exists; skipping %s', input_path)
            return completed
        raise FileExistsError(
            f'output directory contains files from a different or incomplete parser run: '
            f'{paths.recording_dir}. Inspect them, then pass --overwrite to replace only parser outputs.'
        )
    if overwrite:
        _remove_generated_outputs(paths)
    paths.recording_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info('Processing %s', input_path)
    data, info, wear_info, inferred_rate = read_and_process_recording(
        input_path,
        config,
        actipy_module=actipy_module,
        actipy_verbose=actipy_verbose,
    )
    daily_summary = build_daily_summary(data, config)
    eligibility = check_eligibility(data, info, daily_summary, config)

    output_files: list[str] = []
    if eligibility.eligible:
        normalised_dates = data['time'].dt.normalize()
        for summary_index in daily_summary.index[daily_summary['day_ok']]:
            day = pd.Timestamp(daily_summary.loc[summary_index, 'time'])
            day_array = reshape_day(data.loc[normalised_dates == day], config)
            filename = f'day_{len(output_files)}.npy'
            _atomic_npy_dump(day_array, paths.recording_dir / filename)
            daily_summary.loc[summary_index, 'output_file'] = filename
            output_files.append(filename)

    _atomic_csv_dump(daily_summary, paths.wear_duration)
    failed_criteria = [name for name, passed in eligibility.criteria.items() if not passed]
    info.update(
        {
            'interrupts_ok': int(eligibility.criteria['interrupts_ok']),
            'participant_eligibility': int(eligibility.eligible),
            'mean_acceleration_mg': eligibility.mean_acceleration_mg,
            'eligibility': eligibility.criteria,
            'failed_criteria': failed_criteria,
            'nonwear_detection': {
                key: value
                for key, value in wear_info.items()
                if key in {'WearTime(days)', 'NonwearTime(days)', 'NumNonwearEpisodes'}
            },
            'colap_parser': {
                'script_version': SCRIPT_VERSION,
                'schema_version': PARSER_SCHEMA_VERSION,
                'status': 'complete' if eligibility.eligible else 'ineligible',
                'source': source,
                'parameters': config.to_dict(),
                'inferred_input_sample_rate_hz': inferred_rate,
                'eligible_days': int(daily_summary['day_ok'].sum()) if not daily_summary.empty else 0,
                'days_written': len(output_files),
                'output_files': output_files,
                'wear_duration_file': paths.wear_duration.name,
            },
        }
    )
    _atomic_json_dump(info, paths.info)

    if eligibility.eligible:
        LOGGER.info('Wrote %d eligible day(s) to %s', len(output_files), paths.recording_dir)
    else:
        LOGGER.warning('No NPY files written; failed QC: %s', ', '.join(failed_criteria))
    return RunResult(paths.recording_dir, eligibility.eligible, len(output_files))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(levelname)s: %(message)s')
    try:
        result = parse_recording(
            args.file,
            args.output,
            output_name=args.output_name,
            input_sample_rate_hz=args.input_sample_rate,
            overwrite=args.overwrite,
            actipy_verbose=not args.quiet_actipy,
        )
    except Exception as exc:  # CLI boundary: concise by default, traceback in debug mode.
        if args.log_level == 'DEBUG':
            LOGGER.exception('Processing failed')
        else:
            LOGGER.error('%s', exc)
        return EXIT_ERROR
    return EXIT_OK if result.eligible else EXIT_INELIGIBLE


if __name__ == '__main__':
    sys.exit(main())
