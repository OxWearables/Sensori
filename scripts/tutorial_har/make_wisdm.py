"""
Preprocess WISDM data

Raw data:
Sample rate 20Hz
"""

import glob
import os
import zipfile

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

SAMPLE_RATE = 20  # Hz
WINDOW_TOL = 0.01  # 1%
DATASET_URL = "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"


label_dict = {}
label_dict["walking"] = "A"
label_dict["jogging"] = "B"
label_dict["stairs"] = "C"
label_dict["sitting"] = "D"
label_dict["standing"] = "E"
label_dict["typing"] = "F"
label_dict["teeth"] = "G"
label_dict["soup"] = "H"
label_dict["chips"] = "I"
label_dict["pasta"] = "J"
label_dict["drinking"] = "K"
label_dict["sandwich"] = "L"
label_dict["kicking"] = "M"
label_dict["catch"] = "O"
label_dict["dribbling"] = "P"
label_dict["writing"] = "Q"
label_dict["clapping"] = "R"
label_dict["folding"] = "S"
code2name = {code: name for name, code in label_dict.items()}


def post_extract(data_root):
    inner_zip = os.path.join(data_root, "wisdm-dataset.zip")
    if os.path.isfile(inner_zip):
        with zipfile.ZipFile(inner_zip, "r") as zf:
            zf.extractall(data_root)
        os.remove(inner_zip)


def is_good_quality(w, epoch_len):
    """Window quality check"""

    if w.isna().any().any():
        return False

    if len(w) != epoch_len * SAMPLE_RATE:
        return False

    # if len(w['annotation'].unique()) > 1:
    # return False

    w_start, w_end = w.index[0], w.index[-1]
    w_duration = w_end - w_start
    target_duration = pd.Timedelta(epoch_len, "s")
    duration_error = abs(
        w_duration.total_seconds() - target_duration.total_seconds()
    )
    if duration_error > WINDOW_TOL * target_duration.total_seconds():
        return False

    return True


# annolabel = pd.read_csv(ANNOLABELFILE, index_col='annotation')

def process_all(data_root, epoch_len, overlap, save_folder=""):
    all_file_paths = glob.glob(os.path.join(data_root, "wisdm-dataset/raw/watch/accel/*.txt"))

    X, Y, T, P = [], [], [], []

    window_len = epoch_len * SAMPLE_RATE
    step = (epoch_len - overlap) * SAMPLE_RATE
    column_names = ["pid", "code", "time", "x", "y", "z"]

    for datafile in tqdm(all_file_paths):
        one_person_data = pd.read_csv(
            datafile,
            sep=",",
            header=None,
            converters={5: lambda my_x: float(my_x.strip(";"))},
            names=column_names,
        )
        one_person_data["time"] = pd.to_datetime(one_person_data["time"], unit="ns")
        one_person_data = one_person_data.set_index("time")
        period = int(round((1 / SAMPLE_RATE) * 1000_000_000))
        # one_person_data.resample(f'{period}N', origin='start').nearest(limit=1)
        code_to_df = dict(tuple(one_person_data.groupby("code")))
        pid = int(one_person_data["pid"].to_list()[0])

        for code, data in code_to_df.items():
            try:
                data = data.resample(f"{period}ns", origin="start").nearest(limit=1)
            except ValueError:
                if pid == 1629:
                    data = data.drop_duplicates()
                    data = data.resample(f"{period}ns", origin="start").nearest(
                        limit=1
                    )
                    pass

            for i in range(0, len(data), step):
                w = data.iloc[i : i + window_len]

                if not is_good_quality(w, epoch_len):
                    continue

                x = w[["x", "y", "z"]].values
                t = w.index[0].to_datetime64()

                X.append(x)
                Y.append(code2name[code])
                T.append(t)
                P.append(pid)

    X = np.asarray(X)
    Y = np.asarray(Y)
    T = np.asarray(T)
    P = np.asarray(P)

    # fixing unit to g
    X = X / 9.81

    if save_folder == "":
        save_folder = data_root
    
    np.save(os.path.join(save_folder, "X.npy"), X)
    np.save(os.path.join(save_folder, "Y.npy"), Y)
    np.save(os.path.join(save_folder, "time.npy"), T)
    np.save(os.path.join(save_folder, "pid.npy"), P)
    
    print("X shape: ", X.shape)
    print("y shape: ", Y.shape)
    print("pid shape: ", P.shape)
