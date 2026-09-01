"""
Preprocess Capture24 data

Raw data:
Sample rate 100Hz
"""

import glob
import os
import re

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

SAMPLE_RATE = 100  # Hz
WINDOW_TOL = 0.01  # 1%
# LABEL = 'label:Willetts2018'
LABEL = 'label:Walmsley2020'
DATASET_URL = "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/files/rpr76f381b"


def is_good_quality(w, window_sec, window_len):
    ''' Window quality check '''

    if w.isna().any().any():
        return False

    if len(w) != window_len:
        return False

    if len(w['annotation'].unique()) > 1:
        return False

    w_start, w_end = w.index[0], w.index[-1]
    w_duration = w_end - w_start
    target_duration = pd.Timedelta(window_sec, 's')
    if np.abs(w_duration - target_duration) > WINDOW_TOL * target_duration:
        return False

    return True


def process_all(data_root, epoch_len, overlap, save_folder=""):
    window_len = int(SAMPLE_RATE * epoch_len)
    window_overlap_len = int(SAMPLE_RATE * overlap)
    window_step_len = window_len - window_overlap_len

    annolabel = pd.read_csv(
        os.path.join(data_root, "annotation-label-dictionary.csv"),
        index_col='annotation',
    )

    X, Y, T, P = [], [], [], []

    for datafile in tqdm(glob.glob(os.path.join(data_root, "P*.csv.gz"))):
        data = pd.read_csv(
            datafile,
            parse_dates=['time'],
            index_col='time',
            dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'str'},
        )

        p = re.search(r'(P\d{3})', datafile, flags=re.IGNORECASE).group()

        for i in range(0, len(data), window_step_len):
            w = data.iloc[i:i + window_len]

            if not is_good_quality(w, epoch_len, window_len):
                continue

            t = w.index[0].to_datetime64()
            x = w[['x', 'y', 'z']].values
            y = annolabel.loc[w['annotation'].iloc[0], LABEL]

            X.append(x)
            Y.append(y)
            T.append(t)
            P.append(p)

    X = np.asarray(X)
    Y = np.asarray(Y)
    T = np.asarray(T)
    P = np.asarray(P)

    if save_folder == "":
        save_folder = data_root

    os.makedirs(save_folder, exist_ok=True)
    np.save(os.path.join(save_folder, "X.npy"), X)
    np.save(os.path.join(save_folder, "Y.npy"), Y)
    np.save(os.path.join(save_folder, "time.npy"), T)
    np.save(os.path.join(save_folder, "pid.npy"), P)

    print(f"Saved in {save_folder}")
    print("X shape:", X.shape)
    print("Y distribution:")
    print(pd.Series(Y).value_counts())
