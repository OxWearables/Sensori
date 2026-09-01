import glob
import os
import re
from collections import OrderedDict
from typing import Optional

import numpy as np
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

_DAY_FILE = re.compile(r'day_(\d+)\.npy$')
# Preprocessing must save finite float32 arrays with the expected 24-hour shape.


def _find_participant_days(data_root):
    data_root = os.path.abspath(data_root)
    day_files = glob.glob(os.path.join(data_root, '*', 'day_*.npy'))
    day_files.extend(glob.glob(os.path.join(data_root, '*', '*', 'day_*.npy')))

    participant_days = {}
    for day_file in day_files:
        match = _DAY_FILE.fullmatch(os.path.basename(day_file))
        if match is not None:
            participant_days.setdefault(os.path.dirname(day_file), []).append((int(match.group(1)), day_file))

    return [
        (participant_path, [day_file for _, day_file in sorted(days)])
        for participant_path, days in sorted(participant_days.items())
    ]


def _shared_or_private_empty(shape):
    prototype = torch.empty((), dtype=torch.float32)
    if get_worker_info() is None:
        return torch.empty(shape, dtype=torch.float32)
    elements = int(np.prod(shape, dtype=np.int64))
    storage = prototype._typed_storage()._new_shared(elements, device=torch.device('cpu'))
    return prototype.new(storage).resize_(shape)


class DayDataset(Dataset):
    def __init__(self, data_root, sample_hz=10, window_seconds=60):
        """
        Inference dataset that batches day-level .npy files directly from mmap.

        Directory structure:
            data_root/
                participant_1/
                    day_1.npy
                    day_2.npy
                    ...
                participant_2/
                    day_1.npy
                    ...

        Args:
            data_root: Path to root directory containing participant folders
            sample_hz: Samples per second in each day file
            window_seconds: Seconds grouped into each model input window
        """
        self.data_root = os.path.abspath(data_root)
        self.samples_per_window = int(sample_hz * window_seconds)
        self.day_samples = int(24 * 3600 * sample_hz)
        self.samples = []
        self.participant_ids = []

        participant_days = _find_participant_days(self.data_root)
        if not participant_days:
            raise ValueError(f'No day files found in {self.data_root}')

        for participant_index, (participant_path, day_files) in enumerate(participant_days):
            pid = os.path.basename(participant_path)
            self.participant_ids.append(pid)
            self.samples.extend(
                (participant_index, int(_DAY_FILE.fullmatch(os.path.basename(day_file)).group(1)), day_file)
                for day_file in day_files
            )

        print(f'Total days loaded: {len(self.samples)}')
        print(f'Total unique participants: {len(participant_days)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate_fn(self, samples):
        output = _shared_or_private_empty((len(samples), self.day_samples, 3))
        output_array = output.numpy()
        participant_indices = []
        day_ids = []
        for row, (participant_index, day_id, day_path) in enumerate(samples):
            day = np.load(day_path, mmap_mode='r', allow_pickle=False)
            np.copyto(output_array[row], day.reshape(self.day_samples, 3), casting='unsafe')
            participant_indices.append(participant_index)
            day_ids.append(day_id)
        return {
            'data': output.reshape(len(samples), -1, self.samples_per_window, 3),
            'participant_index': torch.tensor(participant_indices),
            'day_id': torch.tensor(day_ids),
        }


class PretrainingDataset(Dataset):
    """Participant index whose collator loads two crop views into one shared batch."""

    def __init__(
        self,
        data_root,
        random_day=True,
        crop_hours=24,
        sample_hz=10,
        min_days_per_participant=6,
        enforce_non_overlap=True,
        max_open_files=16,
    ):
        self.random_day = random_day
        self.sample_hz = int(sample_hz)
        self.crop_hours = int(crop_hours)
        self.crop_len_samples = self.crop_hours * 3600 * self.sample_hz
        self.day_len_samples = 24 * 3600 * self.sample_hz
        self.min_days_per_participant = int(min_days_per_participant)
        self.enforce_non_overlap = enforce_non_overlap
        self.max_open_files = max(1, int(max_open_files))
        self._mmap_cache = OrderedDict()

        roots = [data_root] if isinstance(data_root, (str, os.PathLike)) else list(data_root)
        roots = [os.fspath(root) for root in roots]

        total_days = 0
        self.participants = []
        for root in roots:
            root_subjects = 0
            root_days = 0
            participant_days = _find_participant_days(root)
            if not participant_days:
                raise ValueError(f'No day files found in {root}')

            for participant_path, day_files in participant_days:
                if len(day_files) < self.min_days_per_participant:
                    continue

                self.participants.append((participant_path, day_files))
                root_subjects += 1
                root_days += len(day_files)

            print(f'[{root}] subjects: {root_subjects}, days: {root_days}')
            total_days += root_days

        print(f'Total subjects with valid data: {len(self.participants)}')
        print(f'Total days: {total_days}')

        if not self.participants:
            raise ValueError('No valid participants after filtering by available day files.')

    def _sample_two_starts(self, total_len):
        max_start = total_len - self.crop_len_samples
        if max_start < 0:
            raise ValueError(
                f'Crop length ({self.crop_len_samples}) is larger than available sequence length ({total_len}).'
            )

        if self.enforce_non_overlap and total_len < 2 * self.crop_len_samples:
            raise ValueError(
                f'Cannot sample two non-overlapping crops: total_len={total_len}, crop_len={self.crop_len_samples}'
            )

        if not self.random_day:
            return (0, self.crop_len_samples) if self.enforce_non_overlap else (0, 0)

        if not self.enforce_non_overlap:
            start_0 = np.random.randint(0, max_start + 1)
            start_1 = np.random.randint(0, max_start + 1)
            return start_0, start_1

        crop_len = self.crop_len_samples
        slack = total_len - 2 * crop_len
        if slack >= crop_len:
            start_0 = np.random.randint(0, max_start + 1)
        else:
            edge_width = slack + 1
            first_draw = np.random.randint(0, 2 * edge_width)
            start_0 = first_draw if first_draw < edge_width else crop_len + first_draw - edge_width

        left_count = max(0, start_0 - crop_len + 1)
        right_start = start_0 + crop_len
        right_count = max(0, max_start - right_start + 1)
        second_draw = np.random.randint(0, left_count + right_count)
        start_1 = second_draw if second_draw < left_count else right_start + second_draw - left_count
        return start_0, start_1

    def _open_day(self, day_file):
        day_arr = self._mmap_cache.pop(day_file, None)
        if day_arr is None:
            day_arr = np.load(day_file, mmap_mode='r', allow_pickle=False)
            if day_arr.dtype != np.float32 or day_arr.size != self.day_len_samples * 3:
                raise ValueError(f'Unexpected source array {day_file}: shape={day_arr.shape}, dtype={day_arr.dtype}')
            day_arr = day_arr.reshape(self.day_len_samples, 3)
        self._mmap_cache[day_file] = day_arr
        while len(self._mmap_cache) > self.max_open_files:
            _, evicted = self._mmap_cache.popitem(last=False)
            mapping = getattr(evicted, '_mmap', None)
            if mapping is not None:
                mapping.close()
        return day_arr

    def _fill_week_crop(self, destination, day_files, start):
        copied = 0
        while copied < self.crop_len_samples:
            day_idx, offset = divmod(start + copied, self.day_len_samples)
            day = self._open_day(day_files[day_idx])
            length = min(self.day_len_samples - offset, self.crop_len_samples - copied)
            source = day[offset : offset + length]
            np.copyto(destination[copied : copied + length], source)
            copied += length

    def __len__(self):
        return len(self.participants)

    def __getitem__(self, idx):
        return self.participants[idx]

    def collate_fn(self, participants):
        output = _shared_or_private_empty((2 * len(participants), self.crop_len_samples, 3))
        output_array = output.numpy()
        for row, (_, week_files) in enumerate(participants):
            starts = self._sample_two_starts(len(week_files) * self.day_len_samples)
            for view, start in enumerate(starts):
                output_row = row if view == 0 else len(participants) + row
                self._fill_week_crop(output_array[output_row], week_files, start)
        return output.reshape(2 * len(participants), -1, self.sample_hz * 60, 3)


class PretrainingDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=1,
        num_workers=8,
        crop_hours=24,
        crop_sample_hz=10,
        min_days_per_participant=6,
        enforce_non_overlap=True,
        persistent_workers=False,
        prefetch_factor=None,
        embedding_datasets=None,
        embedding_batch_size=100,
        embedding_num_workers=6,
        embedding_prefetch_factor=2,
    ):
        """
        Data module for participant-level accelerometer pretraining.

        Args:
            data_dir: path to the directory with the data
            batch_size: number of samples per batch
            num_workers: number of workers for data loading
        """

        super().__init__()
        if isinstance(data_dir, (str, os.PathLike)):
            self.data_dir = os.fspath(data_dir)
        else:
            self.data_dir = [os.fspath(path) for path in data_dir]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[list[Dataset]] = None
        self.crop_hours = int(crop_hours)
        self.crop_sample_hz = int(crop_sample_hz)
        self.min_days_per_participant = int(min_days_per_participant)
        self.enforce_non_overlap = enforce_non_overlap
        self.persistent_workers = bool(persistent_workers) and self.num_workers > 0
        self.prefetch_factor = prefetch_factor
        self.embedding_datasets = dict(embedding_datasets or {})
        self.embedding_batch_size = embedding_batch_size
        self.embedding_num_workers = embedding_num_workers
        self.embedding_prefetch_factor = embedding_prefetch_factor

    def setup(self, stage=None):
        if self.data_train is None:
            # Collect all training roots across all data sources and create one
            # dataset, avoiding ConcatDataset index translation during sampling.
            dirs = self.data_dir if isinstance(self.data_dir, list) else [self.data_dir]

            train_roots = []
            for d in dirs:
                train_root = os.path.join(d, 'train') if os.path.exists(os.path.join(d, 'train')) else d
                train_roots.append(train_root)

            self.data_train = PretrainingDataset(
                data_root=train_roots,
                random_day=True,
                crop_hours=self.crop_hours,
                sample_hz=self.crop_sample_hz,
                min_days_per_participant=self.min_days_per_participant,
                enforce_non_overlap=self.enforce_non_overlap,
            )

        if self.data_val is None:
            self.data_val = [DayDataset(data_dir) for data_dir in self.embedding_datasets.values()]

    def train_dataloader(self):
        loader_kwargs = {
            'num_workers': self.num_workers,
            'pin_memory': True,
            'persistent_workers': self.persistent_workers,
        }
        if self.num_workers > 0 and self.prefetch_factor is not None:
            loader_kwargs['prefetch_factor'] = int(self.prefetch_factor)

        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            collate_fn=self.data_train.collate_fn,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )

    def val_dataloader(self):
        if not self.data_val:
            return None

        loader_kwargs = {
            'num_workers': self.embedding_num_workers,
            'pin_memory': True,
        }
        if self.embedding_num_workers > 0 and self.embedding_prefetch_factor is not None:
            loader_kwargs['prefetch_factor'] = int(self.embedding_prefetch_factor)

        return [
            DataLoader(
                dataset,
                batch_size=self.embedding_batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                **loader_kwargs,
            )
            for dataset in self.data_val
        ]
