import argparse
import os
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from tqdm import tqdm

from scripts.tutorial_har import make_capture24, make_pamap2, make_realworld, make_wisdm

DATASET_MODULES = {
    'PAMAP2': make_pamap2,
    'RealWorld': make_realworld,
    'WISDM': make_wisdm,
    'Capture24': make_capture24,
}


def ensure_data(dataset_name, dataset_module, data_root):
    if not os.path.isdir(data_root):
        download_url = str(dataset_module.DATASET_URL)
        if urlparse(download_url).scheme not in {'http', 'https'}:
            raise ValueError(f'Unsupported dataset URL scheme: {download_url!r}')
        zip_path = os.path.join(os.path.dirname(data_root), f'{dataset_name}.zip')
        print(f'Data folder not found. Downloading from {download_url} ...')
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=f'{dataset_name}.zip') as progress:
            downloaded = [0]

            def reporthook(count, block_size, total_size):
                if total_size > 0 and progress.total is None:
                    progress.total = total_size
                progress.update((count - downloaded[0]) * block_size)
                downloaded[0] = count

            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(download_url, zip_path, reporthook=reporthook)  # noqa: S310

        print('Download complete. Extracting...')
        os.makedirs(data_root, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as archive:
            top_level = {path.split('/')[0] for path in archive.namelist() if path.split('/')[0]}
            strip_prefix = (top_level.pop() + '/') if len(top_level) == 1 else ''
            for member in archive.infolist():
                relative_path = member.filename
                if strip_prefix and relative_path.startswith(strip_prefix):
                    relative_path = relative_path[len(strip_prefix) :]
                if not relative_path:
                    continue
                target = os.path.join(data_root, relative_path)
                if member.is_dir():
                    os.makedirs(target, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with archive.open(member) as source, open(target, 'wb') as destination:
                        destination.write(source.read())

        os.remove(zip_path)

        if hasattr(dataset_module, 'post_extract'):
            dataset_module.post_extract(data_root)

        print(f'Extracted to {data_root}')
    else:
        print(f'Data folder found at {data_root}. Skipping download.')

    return data_root


def prepare_datasets(
    data_root,
    datasets=None,
    epoch_len=60,
    overlap=30,
) -> dict[str, Path]:
    """Download and preprocess selected HAR datasets beneath one root directory."""
    data_root = Path(data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    dataset_names = DATASET_MODULES if datasets is None else datasets

    prepared = {}
    for dataset_name in dataset_names:
        dataset_module = DATASET_MODULES[dataset_name]
        dataset_path = data_root / dataset_name
        print(f'Processing dataset: {dataset_name}')
        ensure_data(dataset_name, dataset_module, dataset_path)
        prepared_files = [dataset_path / name for name in ('X.npy', 'Y.npy', 'pid.npy')]
        if all(path.exists() for path in prepared_files):
            print(f'Prepared arrays found at {dataset_path}. Skipping preprocessing.')
        else:
            dataset_module.process_all(dataset_path, epoch_len=epoch_len, overlap=overlap)
        prepared[dataset_name] = dataset_path
    return prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description='Download and preprocess the public HAR datasets.')
    parser.add_argument('data_root', type=Path, help='Directory in which the dataset folders will be created.')
    parser.add_argument('--datasets', nargs='+', choices=DATASET_MODULES, help='Datasets to prepare (default: all).')
    parser.add_argument('--epoch-len', type=int, default=60, help='Window length in seconds (default: 60).')
    parser.add_argument('--overlap', type=int, default=30, help='Window overlap in seconds (default: 30).')
    args = parser.parse_args(argv)
    prepare_datasets(args.data_root, args.datasets, args.epoch_len, args.overlap)


if __name__ == '__main__':
    main()
