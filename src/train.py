from datetime import datetime
from pathlib import Path

import hydra
import pytorch_lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from sensori.datasets import PretrainingDataModule
from sensori.lightning_modules import SensoriPretrainingModule


@hydra.main(version_base=None, config_path='../config', config_name='config_train')
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision('high')
    started_at = datetime.now()
    run_dir = Path(cfg.log_dir) / 'models' / cfg.exp_name / started_at.strftime('%Y-%m-%d_%H_%M_%S')

    wandb_logger = WandbLogger(
        project='sensori',
        name=cfg.exp_name + '_' + started_at.strftime('%Y-%m-%d_%H-%M-%S'),
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wb_mode,
        save_dir=cfg.log_dir,
    )

    pretraining_data = PretrainingDataModule(
        cfg.train_data.data_path,
        batch_size=cfg.train_data.batch_size,
        num_workers=cfg.train_data.num_workers,
        persistent_workers=cfg.train_data.persistent_workers,
        prefetch_factor=cfg.train_data.prefetch_factor,
        embedding_datasets=cfg.test_data.datasets,
        embedding_batch_size=cfg.test_data.batch_size,
        embedding_num_workers=cfg.test_data.num_workers,
        embedding_prefetch_factor=cfg.test_data.prefetch_factor,
    )

    test_enabled = bool(cfg.test_data.datasets)
    embedding_paths = [run_dir / 'embeddings' / f'{name}_epoch_{{epoch}}_embs.npy' for name in cfg.test_data.datasets]
    model = SensoriPretrainingModule(cfg, embedding_paths)

    callbacks = [LearningRateMonitor(logging_interval='step')]

    if cfg.print_model_summary:
        callbacks.append(ModelSummary(max_depth=2))

    if cfg.save_model:
        callbacks.append(
            ModelCheckpoint(
                dirpath=run_dir,
                filename='sensori-{epoch:02d}-{step:05d}',
                save_top_k=-1,
                every_n_epochs=cfg.checkpoint_interval,
            )
        )

    strategy = cfg.strategy
    if cfg.device_num > 1 and cfg.strategy in {'auto', 'ddp'}:
        strategy = DDPStrategy(find_unused_parameters=True)

    trainer = L.Trainer(
        max_epochs=cfg.epoch_len,
        accelerator=cfg.accelerator,
        devices=cfg.device_num,
        strategy=strategy,
        precision=cfg.precision,
        callbacks=callbacks,
        logger=wandb_logger,
        enable_checkpointing=cfg.save_model,
        enable_model_summary=cfg.print_model_summary,
        enable_progress_bar=True,
        log_every_n_steps=cfg.log_step,
        check_val_every_n_epoch=cfg.test_data.every_n_epochs,
        limit_val_batches=1.0 if test_enabled else 0,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=pretraining_data)


if __name__ == '__main__':
    main()
