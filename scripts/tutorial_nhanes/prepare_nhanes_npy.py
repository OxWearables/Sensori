"""Turn one public NHANES participant archive into model-ready daily NPY files.

The CDC publishes one ``.tar.bz2`` archive per participant, holding hourly
``GT3XPLUS-AccelerationCalibrated-*.sensor.csv`` files. ``prepare_participant``
downloads that archive, concatenates the hourly files into the ``time,x,y,z``
CSV that ``scripts/get_npy.py`` reads, runs the parser's fixed preprocessing and
quality-control contract, and deletes the archive and CSV it downloaded.

Everything for one participant lives directly under ``work_dir``::

    work_dir/62161.tar.bz2      deleted unless keep_intermediates is set
    work_dir/62161.csv.gz       deleted unless keep_intermediates is set
    work_dir/62161/day_*.npy    written with info.json and wear_duration.csv

Run one participant from the command line:

    python -m scripts.tutorial_nhanes.prepare_nhanes_npy --eid 62161 --cycle 2011-2012

Exit codes follow ``scripts/get_npy.py``: 0 when eligible days were written,
3 when no day passed quality control, 1 on failure.
"""

import argparse
import gzip
import json
import shutil
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from scripts.get_npy import EXIT_ERROR, EXIT_INELIGIBLE, EXIT_OK, parse_recording

CYCLE_FOLDERS = {'2011-2012': 'pax_g', '2013-2014': 'pax_h'}
# NHANES stores millisecond timestamps, so the 12.5 ms sampling interval alternates
# between 12 and 13 ms gaps. Their mean is 12.5 ms (80 Hz), but their median is 13 ms,
# which is what the parser infers from timestamps: 76.9 Hz. Pass the documented rate.
NOMINAL_SAMPLE_RATE_HZ = 80

OK = 'ok'
NO_ELIGIBLE_DAY = 'no_eligible_day'
FAILED = 'failed'
EXIT_CODES = {OK: EXIT_OK, NO_ELIGIBLE_DAY: EXIT_INELIGIBLE, FAILED: EXIT_ERROR}


@dataclass(frozen=True)
class ParticipantResult:
    """Outcome for one participant. ``status`` is ``ok``, ``no_eligible_day`` or ``failed``."""

    eid: int
    cycle: str
    status: str
    days_written: int = 0
    recording_dir: Path | None = None
    info: dict = field(default_factory=dict)
    message: str = ''

    @property
    def exit_code(self):
        return EXIT_CODES[self.status]

    @property
    def day_files(self):
        if self.recording_dir is None:
            return []
        return sorted(self.recording_dir.glob('day_*.npy'), key=lambda path: int(path.stem.split('_')[1]))

    def load_days(self):
        """Read the written days as ``(2880, 300, 3)`` float32 arrays."""
        return [np.load(path) for path in self.day_files]


def download_archive(eid, cycle, work_dir):
    """Download one participant archive, reusing an existing file."""
    archive_path = Path(work_dir) / f'{eid}.tar.bz2'
    if not archive_path.is_file():
        url = f'https://ftp.cdc.gov/pub/{CYCLE_FOLDERS[cycle]}/{eid}.tar.bz2'
        print(f'Downloading {url}')
        urlretrieve(url, archive_path)
    return archive_path


def combine_hourly_csvs(archive_path, eid, work_dir):
    """Concatenate the archive's hourly files, in time order, into one time,x,y,z CSV."""
    csv_path = Path(work_dir) / f'{eid}.csv.gz'
    if csv_path.is_file():
        return csv_path

    with tarfile.open(archive_path, 'r:bz2') as archive, gzip.open(csv_path, 'wb', compresslevel=6) as output:
        output.write(b'time,x,y,z\n')
        # File names embed the hour, so sorting them puts the samples in time order.
        names = sorted(n for n in archive.getnames() if 'AccelerationCalibrated' in n and n.endswith('.csv'))
        for name in names:
            with archive.extractfile(name) as source:
                source.readline()  # drop the per-file header
                shutil.copyfileobj(source, output)
            output.write(b'\n')  # in case an hourly file has no trailing newline
    return csv_path


def _finished_result(eid, cycle, recording_dir):
    """Rebuild an earlier run's result, so a rerun costs no download."""
    info_path = recording_dir / 'info.json'
    if not info_path.is_file():
        return None
    info = json.loads(info_path.read_text())
    parser_info = info.get('colap_parser', {})
    if parser_info.get('status') not in {'complete', 'ineligible'}:
        return None
    status = OK if parser_info['status'] == 'complete' else NO_ELIGIBLE_DAY
    return ParticipantResult(eid, cycle, status, parser_info.get('days_written', 0), recording_dir, info)


def prepare_participant(eid, cycle, work_dir='.', *, overwrite=False, keep_intermediates=False):
    """Download, combine and preprocess one participant; never raises for a single bad recording.

    Writes ``<work_dir>/<eid>/day_*.npy`` with ``info.json`` and ``wear_duration.csv``,
    then deletes the archive and combined CSV unless ``keep_intermediates`` is set.
    A participant already processed under ``work_dir`` is reported from its existing
    outputs, without downloading anything, unless ``overwrite`` is set. Returns a
    :class:`ParticipantResult`.
    """
    eid = int(eid)
    work_dir = Path(work_dir).expanduser()
    if cycle not in CYCLE_FOLDERS:
        return ParticipantResult(eid, cycle, FAILED, message=f'unknown NHANES cycle: {cycle}')

    if not overwrite:
        finished = _finished_result(eid, cycle, work_dir / str(eid))
        if finished is not None:
            return finished

    work_dir.mkdir(parents=True, exist_ok=True)
    archive_path = csv_path = None
    try:
        archive_path = download_archive(eid, cycle, work_dir)
        csv_path = combine_hourly_csvs(archive_path, eid, work_dir)
        run = parse_recording(
            csv_path,
            work_dir,
            output_name=str(eid),
            input_sample_rate_hz=NOMINAL_SAMPLE_RATE_HZ,
            overwrite=overwrite,
            actipy_verbose=False,
        )
    except (OSError, ValueError, tarfile.TarError) as exc:
        return ParticipantResult(eid, cycle, FAILED, message=f'{type(exc).__name__}: {exc}')
    finally:
        if not keep_intermediates:
            for path in (archive_path, csv_path):
                if path is not None:
                    Path(path).unlink(missing_ok=True)

    info = json.loads((run.recording_dir / 'info.json').read_text())
    status = OK if run.eligible else NO_ELIGIBLE_DAY
    return ParticipantResult(eid, cycle, status, run.days_written, run.recording_dir, info)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Download one public NHANES participant and write model-ready daily NPY files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--eid', type=int, required=True, help='participant SEQN')
    parser.add_argument('--cycle', choices=sorted(CYCLE_FOLDERS), required=True, help='NHANES cycle')
    parser.add_argument('--work-dir', type=Path, default=Path('.'), help='directory holding this participant')
    parser.add_argument('--overwrite', action='store_true', help='reprocess a participant already written here')
    parser.add_argument(
        '--keep-intermediates',
        action='store_true',
        help='keep the downloaded archive and combined CSV',
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = prepare_participant(
        args.eid,
        args.cycle,
        args.work_dir,
        overwrite=args.overwrite,
        keep_intermediates=args.keep_intermediates,
    )
    print(f'{result.eid} ({result.cycle}): {result.status}, {result.days_written} day(s) {result.message}'.strip())
    return result.exit_code


if __name__ == '__main__':
    sys.exit(main())
