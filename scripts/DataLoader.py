import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R


import h5py

import json
import scipy.spatial as sps


class H5Data(Dataset):
    def __init__(
        self,
        num_pts,
        h5fname,
        mapfname,
        seed=None,
        n_samples=None,
        eval_mode=False,
    ):
        self.num_pts = num_pts
        self.indices = np.arange(num_pts)
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self.eval_mode = eval_mode
        assert os.path.isfile(h5fname), h5fname
        self.hf5 = h5py.File(h5fname, "r")

        with open(mapfname, "r") as f:
            self.rev_mapping = json.load(f)
        with open(mapfname.replace("rev_", ""), "r") as f:
            self.mapping = json.load(f)
            self.mapping = {int(k): v for k, v in self.mapping.items()}

        self.structures = tuple(k for k in self.rev_mapping)
        
        self.mapping_for_settings = {
            k: v for k, v in self.mapping.items()
        }

        self.num_outputs = len(self.structures) #+ 1

        if self.n_samples:
            for key in self.rev_mapping:
                assert len(self.hf5[key]) >= n_samples
        else:
            for i, key in enumerate(self.rev_mapping):
                if i == 0:
                    self.n_samples = len(self.hf5[key])
                else:
                    assert self.n_samples == len(self.hf5[key])

    def __del__(self):
        self.hf5.close()

    def __getitem__(self, item):
        category = item // self.n_samples

        index = item - (category * self.n_samples)
        category_label = self.structures[category]
        label = self.rev_mapping[category_label]

        points = self.hf5[category_label][index]
        if not self.eval_mode:
            self.rng.shuffle(self.indices)
            points = points[self.indices]
            points[:, :] = R.random(random_state=self.rng.integers(4294967294)).apply(
                points
            )
        points = torch.from_numpy(points).to(torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return points, label

    def __len__(self):
    	return (len(self.structures)) * self.n_samples


class H5TrainingData(H5Data):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

