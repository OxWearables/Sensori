"""HAR embedding extraction helpers."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import sensori.sensori as sensori_model


# Copied verbatim from actipy.processing.butterfilt in ActiPy 3.3.0:
# https://github.com/OxWearables/actipy
# fmt: off
def butterfilt(x, cutoffs, fs, order=8, axis=0):
    """ Butterworth filter. """
    nyq = 0.5 * fs
    if isinstance(cutoffs, tuple):
        hicut, lowcut = cutoffs
        if hicut > 0:
            if lowcut is not None:
                btype = 'bandpass'
                Wn = (hicut / nyq, lowcut / nyq)
            else:
                btype = 'highpass'
                Wn = hicut / nyq
        else:
            btype = 'lowpass'
            Wn = lowcut / nyq
    else:
        btype = 'lowpass'
        Wn = cutoffs / nyq
    sos = signal.butter(order, Wn, btype=btype, analog=False, output='sos')
    y = signal.sosfiltfilt(sos, x, axis=axis)
    y = y.astype(x.dtype, copy=False)

    return y
# fmt: on


def _load_10hz(data_dir: str) -> np.ndarray:
    path = os.path.join(data_dir, 'X.npy')
    x = np.load(path, mmap_mode='r')
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f'Expected shape [N, T, 3], got {tuple(x.shape)}')

    original_hz = x.shape[1] // 60
    step = original_hz // 10
    out = np.empty((len(x), 600, 3), dtype=np.float32)
    for start in tqdm(range(0, len(x), 1024), desc='5 Hz low-pass -> 10 Hz', leave=False):
        batch = np.asarray(x[start : start + 1024], dtype=np.float32)
        batch = butterfilt(batch, 5, fs=original_hz, order=8, axis=1)
        out[start : start + len(batch)] = batch[:, ::step, :][:, :600]
    return out


def _save_npy(arr, out_dir, dataset_basename):
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'{dataset_basename}.npy')
    np.save(save_path, arr)
    print(f'  Saved {arr.shape} → {save_path}')
    return save_path


def _handcraft_features(xyz, sample_rate: int) -> dict:
    feats = {}
    feats['xMean'], feats['yMean'], feats['zMean'] = np.mean(xyz, axis=0)
    feats['xStd'], feats['yStd'], feats['zStd'] = np.std(xyz, axis=0)
    feats['xRange'], feats['yRange'], feats['zRange'] = np.ptp(xyz, axis=0)
    x, y, z = xyz.T
    with np.errstate(divide='ignore', invalid='ignore'):
        feats['xyCorr'] = np.nan_to_num(np.corrcoef(x, y)[0, 1])
        feats['yzCorr'] = np.nan_to_num(np.corrcoef(y, z)[0, 1])
        feats['zxCorr'] = np.nan_to_num(np.corrcoef(z, x)[0, 1])
    m = np.linalg.norm(xyz, axis=1)
    feats['mean'] = np.mean(m)
    feats['std'] = np.std(m)
    feats['range'] = np.ptp(m)
    feats['mad'] = stats.median_abs_deviation(m)
    feats['enmomean'] = np.mean(np.abs(m - 1))
    if feats['std'] > 0.01:
        feats['skew'] = np.nan_to_num(stats.skew(m))
        feats['kurt'] = np.nan_to_num(stats.kurtosis(m))
    else:
        feats['skew'] = feats['kurt'] = 0
    nperseg = min(3 * sample_rate, len(m))
    noverlap = min(2 * sample_rate, len(m) - 1)
    _, powers = signal.welch(m, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, detrend=False, average='median')
    with np.errstate(divide='ignore', invalid='ignore'):
        feats['pentropy'] = np.nan_to_num(stats.entropy(powers + 1e-16))
    freqs, powers = signal.welch(
        m, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, detrend='constant', average='median'
    )
    peaks, _ = signal.find_peaks(powers)
    if len(peaks) >= 2:
        peak_ranks = np.argsort(powers[peaks])[::-1]
        feats['f1'] = freqs[peaks[peak_ranks[0]]]
        feats['f2'] = freqs[peaks[peak_ranks[1]]]
    elif len(peaks) == 1:
        feats['f1'] = feats['f2'] = freqs[peaks[0]]
    else:
        feats['f1'] = feats['f2'] = 0
    return feats


def extract_handcrafted(data_dir, out_dir, dataset_basename):
    """No additional model dependency is required."""
    x = _load_10hz(data_dir)
    rows = [_handcraft_features(x[i], 10) for i in tqdm(range(len(x)), desc=f'handcrafted {dataset_basename}')]
    feats = pd.DataFrame(rows).to_numpy(dtype=np.float32)
    _save_npy(feats, out_dir, dataset_basename)



class _HARDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = x.astype(np.float32, copy=False)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])


def extract_moment(
    data_dir: str,
    out_dir: str,
    dataset_basename: str,
    device,
):
    """Requires the optional ``momentfm`` package."""
    try:
        from momentfm import MOMENTPipeline
    except ImportError as exc:
        raise ImportError(
            'MOMENT extraction requires momentfm. Install it from the official repository:\n'
            '  pip install git+https://github.com/moment-timeseries-foundation-model/moment.git'
        ) from exc

    x = _load_10hz(data_dir)
    model = MOMENTPipeline.from_pretrained('AutonLab/MOMENT-1-large', model_kwargs={'task_name': 'embedding'})
    model.init()
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(_HARDataset(x), batch_size=32, shuffle=False, num_workers=0)
    split_lengths = (304, 296)
    weights = torch.tensor(split_lengths, device=device).view(1, 2, 1) / 600
    all_emb = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'{dataset_basename} | moment', leave=False):
            inputs, masks = [], []
            for segment in torch.split(batch, split_lengths, dim=1):
                segment = segment.to(device).permute(0, 2, 1)
                pad = 512 - segment.shape[-1]
                inputs.append(F.pad(segment, (pad, 0)))
                masks.append(F.pad(torch.ones(len(segment), segment.shape[-1], device=device), (pad, 0)))
            batch_x = torch.stack(inputs, dim=1).flatten(0, 1)
            input_mask = torch.stack(masks, dim=1).flatten(0, 1)

            embeddings = model(x_enc=batch_x, input_mask=input_mask).embeddings
            embeddings = embeddings.reshape(len(batch), len(split_lengths), -1)
            embeddings = (embeddings * weights).sum(dim=1)
            all_emb.append(embeddings.cpu().numpy())

    emb = np.concatenate(all_emb, axis=0).astype(np.float32, copy=False)
    print(f'  [moment] Extracted {emb.shape} embeddings for {dataset_basename}')
    _save_npy(emb, out_dir, dataset_basename)


def extract_biopm(
    data_dir: str,
    out_dir: str,
    dataset_basename: str,
    biopm_root,
):
    """Please follow [Prithvitarale/biopm](https://github.com/Prithvitarale/biopm)."""
    order_path = os.path.join(os.path.dirname(data_dir), 'biopm_preprocessed', dataset_basename, 'feature_order.npy')

    feature_path = Path(biopm_root).expanduser() / 'features' / f'{dataset_basename}_50mr.npz'
    with np.load(feature_path) as data:
        features = data['features']

    # BioPM loads participant files in sorted order; restore the original X/Y/pid order.
    order = np.load(order_path)
    emb = np.empty_like(features)
    emb[order] = features
    _save_npy(emb, out_dir, dataset_basename)


def extract_chronos2(
    data_dir: str,
    out_dir: str,
    dataset_basename: str,
    device,
):
    """Requires the optional ``chronos-forecasting`` package."""
    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise ImportError(
            'Chronos-2 extraction requires chronos-forecasting:\n  pip install chronos-forecasting'
        ) from exc

    device = torch.device(device)
    x = _load_10hz(data_dir)
    model = Chronos2Pipeline.from_pretrained(
        'amazon/chronos-2',
        device_map=str(device),
        dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
    )

    all_emb = []
    for start in tqdm(
        range(0, len(x), 1024),
        desc=f'{dataset_basename} | chronos2',
        leave=False,
    ):
        batch = np.asarray(x[start : start + 1024]).transpose(0, 2, 1)
        states, _ = model.embed(batch, batch_size=768)

        # Each state is [3 axes, context patches + REG + masked output patch, 768].
        # Retain only observed-signal patches, then average over time and axes.
        pooled = torch.stack([state[:, :-2].float().mean(dim=(0, 1)) for state in states])
        all_emb.append(pooled.numpy())

    emb = np.concatenate(all_emb, axis=0).astype(np.float32, copy=False)
    _save_npy(emb, out_dir, dataset_basename)


def extract_harnet30(
    data_dir: str,
    out_dir: str,
    dataset_basename: str,
    device,
):
    """Requires network access for the Torch Hub model download."""
    x = np.load(os.path.join(data_dir, 'X.npy'), mmap_mode='r')
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f'Expected shape [N, T, 3], got {tuple(x.shape)}')

    t_orig = np.linspace(0, 1, x.shape[1], endpoint=True)
    t_new = np.linspace(0, 1, 1800, endpoint=True)

    model = torch.hub.load('OxWearables/ssl-wearables', 'harnet30', pretrained=True, trust_repo=True)
    model = model.to(device).eval()

    all_emb = []
    with torch.no_grad():
        for start in tqdm(
            range(0, len(x), 256),
            desc=f'{dataset_basename} | linear resize -> harnet30',
            leave=False,
        ):
            end = min(start + 256, len(x))
            batch = interp1d(
                t_orig,
                x[start:end],
                kind='linear',
                axis=1,
                assume_sorted=True,
            )(t_new)
            batch = torch.from_numpy(np.asarray(batch, dtype=np.float32))
            batch = batch.permute(0, 2, 1).to(device)
            embeddings = model.feature_extractor(batch).mean(dim=-1)
            all_emb.append(embeddings.cpu().numpy())

    emb = np.concatenate(all_emb, axis=0).astype(np.float32, copy=False)
    _save_npy(emb, out_dir, dataset_basename)


def extract_sensori(
    data_dir,
    out_dir,
    dataset_basename,
    model,
    device,
    batch_size,
) -> None:
    """Requires a loaded Sensori model."""
    device = torch.device(device)
    model = model.to(device).eval()
    dataloader = DataLoader(_HARDataset(_load_10hz(data_dir)), batch_size=batch_size, shuffle=False, num_workers=0)

    embeddings = []
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc='Sensori embeddings', leave=False):
            output = model(batch.unsqueeze(1).to(device), output='conv_features')
            if output.ndim == 3 and output.shape[1] == 1:
                output = output.squeeze(1)
            embeddings.append(output.cpu().numpy())
    _save_npy(np.concatenate(embeddings, axis=0), out_dir, dataset_basename)


def extract_embeddings(
    method,
    dataset_paths,
    output_root,
    *,
    sensori_model_path=None,
    biopm_root=None,
    local_files_only=False,
    device=None,
    batch_size=256,
) -> dict[str, Path]:
    """Dispatch to the selected extraction method."""

    output_dir = Path(output_root).expanduser() / method
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    output_paths = {name: output_dir / f'{name}.npy' for name in dataset_paths}

    if method == 'sensori' and sensori_model_path is None:
        model = sensori_model.load_pretrained_sensori(
            device=device,
            local_files_only=local_files_only,
        )
    elif method == 'sensori':
        model_config = Path(__file__).resolve().parents[2] / 'config' / 'config_model.yaml'
        model = sensori_model.load_sensori(
            Path(sensori_model_path).expanduser(),
            OmegaConf.load(model_config),
            device,
        )
    elif method == 'biopm' and biopm_root is None:
        raise ValueError('Please follow https://github.com/Prithvitarale/biopm before using BioPM.')

    for dataset_name in dataset_paths:
        data_dir = dataset_paths[dataset_name]
        if method == 'sensori':
            extract_sensori(data_dir, output_dir, dataset_name, model, device, batch_size)
        elif method == 'handcrafted':
            extract_handcrafted(data_dir, output_dir, dataset_name)
        elif method == 'moment':
            extract_moment(data_dir, output_dir, dataset_name, device)
        elif method == 'biopm':
            extract_biopm(data_dir, output_dir, dataset_name, biopm_root)
        elif method == 'chronos2':
            extract_chronos2(data_dir, output_dir, dataset_name, device)
        elif method == 'harnet30':
            extract_harnet30(data_dir, output_dir, dataset_name, device)
        else:
            raise ValueError(f'Unknown extraction method {method!r}')

    return output_paths
