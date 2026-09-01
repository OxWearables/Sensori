import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from umap import UMAP

DATASETS = ('PAMAP2', 'RealWorld', 'WISDM', 'Capture24')

UMAP_N_NEIGHBORS = 50
UMAP_MIN_DIST = 0.3
UMAP_SUBSAMPLE_N = 5000
UMAP_RANDOM_SEED = 42

UMAP_DISPLAY_CONFIG = {
    'PAMAP2': {
        'relabel': {
            1: 'lying down',
            2: 'sitting',
            3: 'standing',
            4: 'walking',
            12: 'ascending stairs',
            13: 'descending stairs',
            16: 'vacuum cleaning',
            17: 'ironing',
        },
    },
    'RealWorld': {
        'relabel': {
            'lying': 'lying down',
            'climbingup': 'climbing up',
            'climbingdown': 'climbing down',
        },
    },
    'WISDM': {
        'relabel': {
            'dribbling': 'dribbling basketball',
            'kicking': 'kicking soccer ball',
            'catch': 'catch tennis ball',
            'pasta': 'eating pasta',
            'sandwich': 'eating sandwich',
            'chips': 'eating chips',
            'soup': 'eating soup',
            'drinking': 'drinking from cup',
            'teeth': 'brushing teeth',
            'folding': 'folding clothes',
        },
    },
    'Capture24': {
        'relabel': {},
    },
}


def load_umap_inputs(
    dataset_name: str,
    data_root: str | Path,
    embedding_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    emb_path = Path(embedding_dir).expanduser() / f'{dataset_name}.npy'
    label_path = Path(data_root).expanduser() / dataset_name / 'Y.npy'
    if not emb_path.exists():
        raise FileNotFoundError(f'Missing embedding file: {emb_path}')
    if not label_path.exists():
        raise FileNotFoundError(f'Missing label file: {label_path}')

    x = np.load(emb_path, allow_pickle=True)
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    if x.ndim != 2:
        raise ValueError(f'Expected embedding shape [N, F], got {tuple(x.shape)} from {emb_path}')

    y_raw = np.load(label_path, allow_pickle=True)
    if len(x) != len(y_raw):
        raise ValueError(f'Length mismatch for {dataset_name}: embeddings={len(x)}, labels={len(y_raw)}')

    sample_index = np.arange(len(x))
    if UMAP_SUBSAMPLE_N is not None and len(x) > UMAP_SUBSAMPLE_N:
        original_n = len(x)
        rng = np.random.default_rng(UMAP_RANDOM_SEED)
        sample_index = rng.choice(len(x), size=UMAP_SUBSAMPLE_N, replace=False)
        sample_index.sort()
        x = x[sample_index]
        y_raw = y_raw[sample_index]
        print(f'{dataset_name}: subsampled to {UMAP_SUBSAMPLE_N:,} points from {original_n:,}')

    labels = np.array(
        [UMAP_DISPLAY_CONFIG[dataset_name]['relabel'].get(label, label) for label in y_raw],
        dtype=object,
    )
    return x.astype(np.float32, copy=False), labels, sample_index


def compute_umap_table(
    dataset_name: str,
    data_root: str | Path,
    embedding_dir: str | Path,
) -> pd.DataFrame:
    x, labels, sample_index = load_umap_inputs(dataset_name, data_root, embedding_dir)
    print(f'{dataset_name}: fitting UMAP on {len(x):,} samples')
    projection = UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=UMAP_RANDOM_SEED,
        n_jobs=1,
    ).fit_transform(x)

    return pd.DataFrame(
        {
            'dataset': dataset_name,
            'sample_index': sample_index,
            'label': labels.astype(str),
            'umap_1': projection[:, 0],
            'umap_2': projection[:, 1],
        }
    )


def plot_umap_table(table: pd.DataFrame, ax=None, title: str | None = None):
    """Plot a UMAP table."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    labels = sorted(table['label'].unique())
    colors = plt.get_cmap('tab20', len(labels))
    for index, label in enumerate(labels):
        points = table[table['label'] == label]
        ax.scatter(
            points['umap_1'],
            points['umap_2'],
            s=8,
            alpha=0.7,
            color=colors(index),
            label=label,
            linewidths=0,
            rasterized=True,
        )

    ax.set(xlabel='UMAP 1', ylabel='UMAP 2')
    if title is not None:
        ax.set_title(title)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=2)
    ax.figure.tight_layout()
    return ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create UMAP source tables from HAR embeddings.')
    parser.add_argument('data_root', type=Path)
    parser.add_argument('embedding_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name in DATASETS if args.datasets is None else args.datasets:
        table = compute_umap_table(dataset_name, args.data_root, args.embedding_dir)
        out_path = output_dir / f'{dataset_name}_umap.csv'
        table.to_csv(out_path, index=False)
        print(f'Saved {out_path}')
