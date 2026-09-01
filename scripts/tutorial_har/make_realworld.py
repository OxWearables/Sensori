"""
Preprocess RealWorld2016 data

Raw data:
Sample rate 50Hz
"""

import glob
import os
from zipfile import ZipFile

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

SAMPLE_RATE = 50
DATASET_URL = "http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip"

SUB_IDS = list(range(1, 16))
LABELS = [
    "climbingdown",
    "climbingup",
    "jumping",
    "lying",
    "running",
    "sitting",
    "standing",
    "walking",
]
BODY_PARTS = ["chest", "forearm", "head", "shin", "thigh", "upperarm", "waist"]


def _cal_min_time_len(readme):
    time_len_list = []
    for line in readme.readlines():
        if line.startswith(b"> entries"):
            time_len_list.append(int(line[11:16]))
    return min(time_len_list)


def _cal_xyz(zip_file, csv, min_time_len):
    df = pd.read_csv(zip_file.open(csv))
    xyz = np.stack([df["attr_x"], df["attr_y"], df["attr_z"]], axis=1)
    return xyz[:min_time_len]


def _cal_xyz_acc(zip_file, i, sub_id, label):
    min_time_len = _cal_min_time_len(zip_file.open("readMe"))
    xyz_acc = None
    for b_part in BODY_PARTS:
        if i == 1:
            special = (
                (sub_id == 4 and label == "walking")
                or (sub_id == 6 and label == "sitting")
                or (sub_id == 7 and label == "sitting")
                or (sub_id == 8 and label == "standing")
                or (sub_id == 13 and label == "walking")
            )
            csv = "acc_{}_2_{}.csv".format(label, b_part) if special else "acc_{}_{}.csv".format(label, b_part)
        else:
            csv = "acc_{}_{}_{}.csv".format(label, i, b_part)

        if csv in zip_file.namelist():
            xyz = _cal_xyz(zip_file, csv, min_time_len)
        else:
            print("data missing: {}/{}; missing part: {}".format(sub_id, label, b_part))
            xyz = np.full([min_time_len, 3], np.nan)

        xyz_acc = xyz if xyz_acc is None else np.concatenate((xyz_acc, xyz), axis=1)
    return xyz_acc


def post_extract(data_root):
    """Extract per-subject nested zips into flat .npy files under data_root/imu/."""
    imu_path = os.path.join(data_root, "imu")
    os.makedirs(imu_path, exist_ok=True)

    for sub_id in tqdm(SUB_IDS, desc="Subjects"):
        subject_data_path = os.path.join(data_root, "proband{}", "data").format(sub_id)
        for label in LABELS:
            ziplabel = label.replace("_", "")
            zip_file_path = os.path.join(subject_data_path, "acc_{}_csv.zip".format(ziplabel))
            z_file = ZipFile(zip_file_path)
            if z_file.namelist()[0].endswith("zip"):
                save_zip_path = os.path.join(subject_data_path, "acc_{}_csv".format(ziplabel))
                z_file.extractall(path=save_zip_path)
                for i in range(1, len(z_file.namelist()) + 1):
                    sub_z_file = ZipFile(os.path.join(save_zip_path, "acc_{}_{}_csv.zip".format(ziplabel, i)))
                    xyz_acc = _cal_xyz_acc(sub_z_file, i, sub_id, ziplabel)
                    np.save(os.path.join(imu_path, "{}.{}.{}.npy".format(sub_id, i - 1, label)), xyz_acc)
                    sub_z_file.close()
            else:
                xyz_acc = _cal_xyz_acc(z_file, 1, sub_id, ziplabel)
                np.save(os.path.join(imu_path, "{}.0.{}.npy".format(sub_id, label)), xyz_acc)
            z_file.close()


def _is_good_quality(window, window_len):
    return not np.isnan(window).any() and len(window) == window_len


def process_all(data_root, epoch_len, overlap, target_hz=None, save_folder=""):
    file_paths = glob.glob(os.path.join(data_root, "imu", "*.npy"))

    X, Y, S, P = [], [], [], []
    window_len = epoch_len * SAMPLE_RATE
    step = (epoch_len - overlap) * SAMPLE_RATE

    for datafile in tqdm(file_paths):
        pid, sess_id, class_name, _ = os.path.basename(datafile).split(".")
        data = np.load(datafile)[:, 3:6]  # forearm columns

        for i in range(0, len(data), step):
            window = data[i : i + window_len]
            if not _is_good_quality(window, window_len):
                continue
            X.append(window)
            Y.append(class_name)
            S.append(sess_id)
            P.append(pid)

    X = np.asarray(X)
    Y = np.asarray(Y)
    S = np.asarray(S)
    P = np.asarray(P)

    X = X / 9.81  # convert to g

    if save_folder == "":
        save_folder = data_root
    
    os.makedirs(save_folder, exist_ok=True)
    np.save(os.path.join(save_folder, "X.npy"), X)
    np.save(os.path.join(save_folder, "Y.npy"), Y)
    np.save(os.path.join(save_folder, "session_id.npy"), S)
    np.save(os.path.join(save_folder, "pid.npy"), P)

    print("X shape: ", X.shape)
    print("Y shape: ", Y.shape)
    print("pid shape: ", P.shape)
