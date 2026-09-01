"""Extract day embeddings from preprocessed accelerometer data."""

import argparse
from pathlib import Path

import pytorch_lightning as L
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from sensori.datasets import DayDataset
from sensori.lightning_modules import SensoriEmbeddingModule
from sensori.sensori import load_pretrained_sensori, load_sensori


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Extract one Sensori embedding per day and group them by participant.',
    )
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--output-path', type=Path)
    # Pass both local paths to override the Hugging Face model; omit both to download it.
    parser.add_argument(
        '--checkpoint-path',
        type=Path,
        help='Local checkpoint; must be provided together with --config-path.',
    )
    parser.add_argument(
        '--config-path',
        type=Path,
        help='Model configuration for --checkpoint-path; both paths are required for local loading.',
    )
    parser.add_argument('--accelerator', choices=('auto', 'cpu', 'gpu'), default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=6)
    args = parser.parse_args(argv)
    if (args.checkpoint_path is None) != (args.config_path is None):
        parser.error('--checkpoint-path and --config-path must be provided together.')
    return args


def extract_embeddings(
    data_path,
    output_path=None,
    *,
    checkpoint_path=None,
    config_path=None,
    accelerator='auto',
    devices=1,
    batch_size=100,
    num_workers=6,
):
    """Extract embeddings using paired local model paths or the Hugging Face release."""
    if (checkpoint_path is None) != (config_path is None):
        raise ValueError('checkpoint_path and config_path must be provided together.')

    data_path = Path(data_path).expanduser().resolve()
    if output_path is None:
        save_path = data_path.parent / 'sensori_embeddings' / f'{data_path.name}_embs.npy'
    else:
        save_path = Path(output_path).expanduser().resolve()
        if save_path.suffix != '.npy':
            save_path = Path(f'{save_path}.npy')

    if save_path.exists():
        print(f'Embeddings already exist: {save_path}')
        return save_path

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision('high')

    if checkpoint_path is None:
        model = load_pretrained_sensori()
    else:
        cfg = OmegaConf.load(Path(config_path).expanduser().resolve())
        model = load_sensori(Path(checkpoint_path).expanduser().resolve(), cfg)
    dataset = DayDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    print(f'Loading data from: {data_path}')
    print(f'Saving embeddings to: {save_path}')

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy='auto',
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )
    trainer.validate(
        SensoriEmbeddingModule(model, [save_path]),
        dataloaders=[dataloader],
        verbose=False,
    )
    return save_path


def main(argv=None):
    extract_embeddings(**vars(parse_args(argv)))


if __name__ == '__main__':
    main()
