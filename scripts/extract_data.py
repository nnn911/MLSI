import glob
from ovito.io import import_file,export_file
import numpy as np
import scipy.spatial as sps
import os
from tqdm import trange, tqdm
import json
from ovito.modifiers import ReplicateModifier, ExpressionSelectionModifier, CombineDatasetsModifier
from ovito.data import NearestNeighborFinder
from ovito.pipeline import StaticSource, Pipeline
import sys
import yaml


class OVITO_dataset:
    def __init__(self, settings: dict):
        self.settings = settings
        self.rng = np.random.default_rng(settings["seed"])
        pipeline = import_file(self.settings["fname"])
        self.pipe = pipeline
        print("Number of Files: ",self.pipe.source.num_frames)
        self.max_frames = self._find_max_frames(2)
        os.system("mkdir -p "+self.settings["odir"])

    def _find_max_frames(self, factor):
        v0 = self.pipe.compute(0).cell.volume
        for i in range(self.pipe.source.num_frames):
            data = self.pipe.compute(i)
            if data.cell.volume / v0 > factor:
                return i
        return self.pipe.source.num_frames

    def _get_structure_from_data(self, data, index, finder):
        ret = np.zeros((self.settings["n_atoms"], 3), dtype=self.settings["dtype"])
        for i, neigh in enumerate(finder.find(index)):
            ret[i] = neigh.delta
        ret /= np.linalg.norm(ret[0])
        assert np.isclose(np.min(np.linalg.norm(ret, axis=1)), 1)
        return ret

    def _get_uniform_indices(self, data, n_samples):
        ptypes = np.unique(data.particles["Particle Type"])
        indices = []
        for p in ptypes:
            indices.append(
                self.rng.choice(
                    np.where((data.particles["Particle Type"] == int(p)))[0],
                    size=n_samples // len(ptypes),
                )
            )
        for i in range(n_samples - np.sum([len(i) for i in indices])):
            indices.append(
                self.rng.choice(
                    np.where(
                        (
                            data.particles["Particle Type"]
                            == int(ptypes[i % len(ptypes)])
                        )
                    )[0],
                    size=1,
                )
            )
        ret = np.hstack(indices)
        self.rng.shuffle(ret)
        # assert len(ret) == n_samples
        return ret

    def _get_structures_from_data(self, data, n_samples):
        finder = NearestNeighborFinder(self.settings["n_atoms"], data)
        if self.settings["uniform"]:
            indices = self._get_uniform_indices(data, n_samples)
        else:
            indices = self.rng.choice(np.arange(data.particles.count), size=n_samples)
        ret = np.zeros(
            (n_samples, self.settings["n_atoms"], 3), dtype=self.settings["dtype"]
        )
        for i, index in enumerate(tqdm(indices, leave=False)):
            ret[i, :, :] = self._get_structure_from_data(data, index, finder)
        return ret

    def get_structures(self):
        n_samples = self.settings["n_samples"] // self.max_frames
        ret = np.zeros(
            (self.settings["n_samples"], self.settings["n_atoms"], 3),
            dtype=self.settings["dtype"],
        )
        if n_samples > 0:
            for frame in trange(self.max_frames, leave=False):
                data = self.pipe.compute(frame)
                ret[
                    frame * n_samples : (frame + 1) * n_samples,
                    :,
                    :,
                ] = self._get_structures_from_data(data, n_samples)
        for frame in trange(self.settings["n_samples"] % self.max_frames, leave=False):
            data = self.pipe.compute(self.rng.integers(0, self.max_frames))
            ret[
                self.max_frames * n_samples + frame, :, :
            ] = self._get_structures_from_data(data, 1)
        return ret

    def export_structures(self):
        strucs = self.get_structures()
        np.save(self.settings["oname"], strucs)
        self.settings["dtype"] = str(self.settings["dtype"])
        with open(self.settings["oname"].replace("npy", "json"), "w") as f:
            json.dump(self.settings, f, sort_keys=True, indent="  ")


def construct_settings(settings):
    settings["n_samples"] = int(settings["n_samples"])
    settings["n_atoms"] = int(settings["n_atoms"])
    settings["seed"] = int(settings["seed"])
    settings["uniform"] = bool(settings["uniform"])

    filenames = []

    usetxt=False
    
    files = os.listdir(settings["prepath"]+settings["fdir"])

    for f in files:
        filenames.append(os.path.join(
                prepath, settings["phase"], f
        ))


    settings["fname"] = filenames
    settings["oname"] = os.path.join(
        settings["odir"],
        f"{settings['oname_prefix']}_{settings['phase']}_{settings['n_samples']}_{settings['n_atoms']}.npy",
    )
    return settings


def run_single(settings):
    dataset = OVITO_dataset(settings)
    dataset.export_structures()


def get_settings(structurename,points,prepath):
    settings = []
    settings.append(
        construct_settings(
            {
               "prepath": prepath,
               "fdir": structurename,
               "phase": structurename,
               "odir": "1000000_"+str(points),
               "oname_prefix": "training",
               "uniform": True,
               "seed": 123456,
               "n_samples": 1e6,
               "n_atoms": int(points),
               "dtype": np.single,
            }
        )
    )
    return settings


def main(structurename,points,prepath):
    settings = get_settings(structurename,points,prepath)
    run_single(settings[0])



if __name__ == "__main__":
    prepath = os.getcwd()+"/md_training_data/"
    main(sys.argv[1],int(sys.argv[2]),prepath)

