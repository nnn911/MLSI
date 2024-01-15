import glob
import numpy as np
import os
from tqdm import trange, tqdm
import json
import h5py
import sys



def _write(npy_files, h5file, mapping, rev_mapping):
    start = 0
    if len(mapping.keys()) > 0:
        start = max(mapping.keys()) + 1
    for i, npy in enumerate(npy_files):
        with open(npy.replace("npy", "json"), "r") as f:
            settings = json.load(f)
        struc = settings["phase"]
        if struc not in rev_mapping:
            mapping[i + start] = struc
            rev_mapping[struc] = i + start
        data = np.load(npy)
        initial_shape = data.shape
        rng = np.random.default_rng(123456)
        rng.shuffle(data, axis=0)
        assert data.shape == initial_shape
        print(struc, data.shape)
        h5file.create_dataset(
            struc,
            data=data,
        )


def write(baseName):
    mapping = {}
    rev_mapping = {}

    h5name = f"numpy_data.h5"
    with h5py.File(os.path.join(baseName, h5name), "w") as hf:
        npy_files = glob.glob(os.path.join(baseName, "*1000000_*.npy"))
        print(npy_files)
        npy_files.sort()
        _write(npy_files, hf, mapping, rev_mapping)

    with open(os.path.join(baseName, "mapping.json"), "w") as f:
        json.dump(mapping, f)
    with open(os.path.join(baseName, "rev_mapping.json"), "w") as f:
        json.dump(rev_mapping, f)



if __name__ == "__main__":
    write(sys.argv[1])
