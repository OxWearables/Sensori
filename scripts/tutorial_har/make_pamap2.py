"""
Preprocess PAMAP2 data

Raw data:
Range +- 16g
Sample rate 100Hz
"""

import glob
import os

import numpy as np
from scipy import constants
from scipy import stats as s
from tqdm import tqdm

SAMPLE_RATE = 100
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"


def get_data_content(data_path):
    # read flash.dat to a list of lists
    datContent = [i.strip().split() for i in open(data_path).readlines()]
    datContent = np.array(datContent)

    label_idx = 1
    timestamp_idx = 0
    x_idx = 4
    y_idx = 5
    z_idx = 6
    index_to_keep = [timestamp_idx, label_idx, x_idx, y_idx, z_idx]
    # 3d +- 16 g

    datContent = datContent[:, index_to_keep]
    datContent = datContent.astype(float)
    datContent = datContent[~np.isnan(datContent).any(axis=1)]
    return datContent


def content2x_and_y(data_content, epoch_len=30, overlap=15):
    sample_label_idx = 1
    sample_x_idx = 2
    sample_y_idx = 3
    sample_z_idx = 4

    window_len = epoch_len * SAMPLE_RATE
    step = (epoch_len - overlap) * SAMPLE_RATE

    X, label = [], []
    for i in range(0, len(data_content) - window_len + 1, step):
        w = data_content[i : i + window_len]
        label.append(w[:, sample_label_idx])
        X.append(w[:, [sample_x_idx, sample_y_idx, sample_z_idx]])

    return np.array(X), np.array(label)


def clean_up_label(X, labels):
    # 1. remove rows with >50% zeros
    sample_count_per_row = labels.shape[1]

    rows2keep = np.ones(labels.shape[0], dtype=bool)
    transition_class = 0
    for i in range(labels.shape[0]):
        row = labels[i, :]
        if np.sum(row == transition_class) > 0.5 * sample_count_per_row:
            rows2keep[i] = False

    labels = labels[rows2keep]
    X = X[rows2keep]

    # 2. majority voting for label in each epoch
    final_labels = []
    purity = []
    for i in range(labels.shape[0]):
        row = labels[i, :]
        majority = s.mode(row)[0]
        final_labels.append(majority)
        purity.append(np.sum(row == majority) / len(row))
    final_labels = np.array(final_labels, dtype=int)
    purity = np.array(purity)
    return X, final_labels, purity


def process_all(data_root, epoch_len, overlap, save_folder=""):
    protocol_file_paths = glob.glob(os.path.join(data_root, "Protocol", "*.dat"))
    optional_file_paths = glob.glob(os.path.join(data_root, "Optional", "*.dat"))
    all_file_paths = protocol_file_paths + optional_file_paths
    
    X, y, pid, all_purity = [], [], [], []
    
    for file_path in tqdm(all_file_paths):
        subject_id = int(file_path.split("/")[-1][-7:-4])
        datContent = get_data_content(file_path)
        current_X, current_y = content2x_and_y(
            datContent, epoch_len=epoch_len, overlap=overlap
        )
        current_X, current_y, purity = clean_up_label(current_X, current_y)
        all_purity.append(purity)
        ids = np.full(
            shape=len(current_y), fill_value=subject_id, dtype=np.int32
        )
        if len(X) == 0:
            X = current_X
            y = current_y
            pid = ids
        else:
            X = np.concatenate([X, current_X])
            y = np.concatenate([y, current_y])
            pid = np.concatenate([pid, ids])

    y = y.flatten()
    X = X / constants.g  # convert to unit of g
    clip_value = 3
    X = np.clip(X, -clip_value, clip_value)

    # Keep only 8 activities that everyone has
    y_filter = (
        (y == 1)
        | (y == 2)
        | (y == 3)
        | (y == 4)
        | (y == 12)
        | (y == 13)
        | (y == 16)
        | (y == 17)
    )
    X = X[y_filter]
    y = y[y_filter]
    pid = pid[y_filter]

    if save_folder == "":
        save_folder = data_root
    
    os.makedirs(save_folder, exist_ok=True)    
    np.save(os.path.join(save_folder, "X.npy"), X)
    np.save(os.path.join(save_folder, "Y.npy"), y)
    np.save(os.path.join(save_folder, "pid.npy"), pid)
    
    all_purity = np.concatenate(all_purity)
    print(f"Label purity — mean: {all_purity.mean():.3f}, "
          f"min: {all_purity.min():.3f}, "
          f"median: {np.median(all_purity):.3f}, "
          f"100%: {(all_purity == 1.0).mean()*100:.1f}% of windows")
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    print("pid shape: ", pid.shape)
