import importlib.resources as impRes
import json
import os
import time

import numpy as np
import torch
from ovito.data import DataCollection, DataTable, ElementType
from ovito.modifiers import (
    DeleteSelectedModifier,
    ExpressionSelectionModifier,
    ReplicateModifier,
)
from ovito.pipeline import ModifierInterface
from torch.utils.data import DataLoader
from traits.api import Bool, Range, String

from .DG_CNN_tools import (
    OvitoDataSet_KDTree,
    OvitoDataSet_NearestNeighborFinder,
    Restored_Model,
)


class MLSI(ModifierInterface):
    model_dir = String("", label="Model directory")
    cutoff = Range(0.0, None, value=0.0, label="Cutoff")
    use_neighbor_finder = Bool(True, label="Use: Nearest Neighbor Finder")
    use_KDTree = Bool(False, label="Use: KDTree")
    rep = Range(1, None, value=1, label="Replicate cell")
    buffer = Range(0.0, 1.0, value=0.1, label="Thickness of buffer zone")

    def modify(self, data: DataCollection, **kwargs):
        start = time.perf_counter()

        if (self.use_neighbor_finder + self.use_KDTree) != 1:
            raise ValueError(
                "Select only one option of: NearestNeighborFinder and KDTree!"
            )
        if (not isinstance(self.model_dir, str)) or (len(self.model_dir) == 0):
            print("---------------------------------")
            print(
                f"Pretrained models can be found at: {impRes.files('MLSI.pretrained_models').joinpath('')}"
            )
            print("---------------------------------")
            raise ValueError(
                f"Model directory: '{self.model_dir}' is empty or not a valid directory name!"
            )
        if not os.path.isdir(self.model_dir):
            print("---------------------------------")
            print(
                f"Pretrained models can be found at: {impRes.files('MLSI.pretrained_models').joinpath('')}"
            )
            print("---------------------------------")
            raise NotADirectoryError(
                f"Model directory: {self.model_dir} does not exist!"
            )
        if self.KDTree and (self.rep % 2) == 0:
            raise ValueError("rep needs to be an odd number!")

        with open(os.path.join(self.model_dir, "settings.json"), "r") as f:
            settings = json.load(f)

        mapping = {int(k): v for k, v in settings["classes"].items()}
        mapping[-1] = "Other"
        print("Possible classes:")
        for k, v in mapping.items():
            print(f"{k}: {v}")

        torch.manual_seed(settings["seed"])
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        model = Restored_Model(self.model_dir)

        batch_size = 8 * settings["batch_size"]
        if self.use_KDTree:
            original_num_atoms = data.particles.count
            rep = np.array((self.rep, self.rep, self.rep), dtype=int) * data.cell.pbc
            data.apply(
                ReplicateModifier(
                    num_x=rep[0], num_y=rep[1], num_z=rep[2], adjust_box=False
                )
            )
            expr = (
                f"ReducedPosition.X <= {-self.buffer} || ReducedPosition.X > {1+self.buffer}"
                f"|| ReducedPosition.Y <= {-self.buffer} || ReducedPosition.Y > {1+self.buffer}"
                f"|| ReducedPosition.Z <= {-self.buffer} || ReducedPosition.Z > {1+self.buffer}"
            )
            data.apply(ExpressionSelectionModifier(expression=expr))
            data.apply(DeleteSelectedModifier())
            dataSet = OvitoDataSet_KDTree(
                data.particles["Position"], settings["num_pts"]
            )
        else:
            dataSet = OvitoDataSet_NearestNeighborFinder(data, settings["num_pts"])

        struc = np.zeros(data.particles.count, dtype=int)
        score = np.zeros(data.particles.count)

        batch_size = 8 * settings["batch_size"]
        loader = DataLoader(dataSet, num_workers=0, batch_size=batch_size)

        for i, dat in enumerate(loader):
            dat = dat.permute(0, 2, 1)
            dat = dat.float().to(device)
            pred = model(dat).softmax(dim=1)
            val, ind = torch.max(pred, 1)
            struc[i * batch_size : (i + 1) * batch_size] = ind.cpu()
            score[i * batch_size : (i + 1) * batch_size] = val.cpu()
            yield i / len(loader)

        struc[score < self.cutoff] = -1
        data.particles_.create_property("Structure", data=struc)
        data.particles_.create_property("Score", data=score)

        if self.use_KDTree:
            expr = (
                r"ReducedPosition.X < 0 || ReducedPosition.X >= 1"
                r"|| ReducedPosition.Y < 0 || ReducedPosition.Y >= 1"
                r"|| ReducedPosition.Z < 0 || ReducedPosition.Z >= 1"
            )
            data.apply(ExpressionSelectionModifier(expression=expr))
            data.apply(DeleteSelectedModifier())
            if data.particles.count != original_num_atoms:
                print(
                    "Warning: Atoms gained or lost during padding! Proceed with caution or use NearestNeighborFinder instead of KDTree"
                )

        uni, cts = np.unique(data.particles["Structure"], return_counts=True)

        print("\nClassification result:")
        for u, c in zip(uni, cts):
            print(mapping[int(u)], f"{c / data.particles.count :.3f}")
            data.attributes[f"MLSI.counts.{mapping[int(u)]}"] = c

        table = DataTable(
            title="Structure counts",
            plot_mode=DataTable.PlotMode.BarChart,
        )
        table.x = table.create_property("Structure type", data=uni)
        for i, (u, c) in enumerate(zip(uni, cts)):
            table.x.types.append(ElementType(id=i, name=mapping[u]))
        table.y = table.create_property("Count", data=cts)
        data.objects.append(table)

        print(f"\nWall time: {time.perf_counter() - start} s")
