import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ovito.data import DataCollection, NearestNeighborFinder, DataTable, ElementType
from ovito.modifiers import (
    DeleteSelectedModifier,
    ExpressionSelectionModifier,
    ReplicateModifier,
)
from torch.utils.data import DataLoader, Dataset


# Copyright for parts of DGCNN_cls are held by An Tao, 2020 as part of
# GCNN.pytorch (https://github.com/antao97/dgcnn.pytorch) and are provided under
# the MIT license.
class DGCNN_cls(nn.Module):
    @staticmethod
    def knn(x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx

    @staticmethod
    def get_graph_feature(x, settings, idx=None):
        k = settings["k"]
        dim9 = settings["dim9"]
        device = settings["device"]

        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            if dim9 == False:
                idx = DGCNN_cls.knn(x, k=k)
            else:
                idx = DGCNN_cls.knn(x[:, 6:], k=k)
        idx_base = (
            torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        )
        idx = idx + idx_base
        idx = idx.view(-1)
        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature

    def __init__(self, settings, output_channels):
        super(DGCNN_cls, self).__init__()
        self.settings = settings

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.settings["emb_dims"])

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, self.settings["emb_dims"], kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.linear1 = nn.Linear(self.settings["emb_dims"] * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.settings["dropout"])
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.settings["dropout"])
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.get_graph_feature(x, self.settings)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = self.get_graph_feature(x1, self.settings)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = self.get_graph_feature(x2, self.settings)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = self.get_graph_feature(x3, self.settings)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class Restored_Model:
    def __init__(self, model_dir=None):
        if not model_dir:
            print(f"{model_dir} is not a valid directory name!")
            return
        if not os.path.isdir(model_dir):
            print(f"{model_dir} does not exist!")
            return
        with open(os.path.join(model_dir, "settings.json"), "r") as f:
            self.settings = json.load(f)

        self.mapping = self.settings["classes"]
        self.mapping = {int(k): v for k, v in self.mapping.items()}
        self.mapping[-1] = ["other"]

        torch.manual_seed(self.settings["seed"])
        self.settings["device"] = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.model = DGCNN_cls(self.settings, self.settings["num_outputs"])
        model_fname = "torch_checkpoint_best.tar"
        state_dict = torch.load(
            os.path.join(model_dir, model_fname), map_location=self.settings["device"]
        )["model_state_dict"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict, prefix="module."
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.settings["device"])

    def __call__(self, *args, **kwds):
        self.model.eval()
        with torch.no_grad():
            return self.model(*args, **kwds)


class OvitoDataSet_NearestNeighborFinder(Dataset):
    def __init__(self, data, num_pts):
        self.num_pts = num_pts
        self.particle_count = data.particles.count
        try:
            self.finder = NearestNeighborFinder(num_pts, data)
        except ValueError as e:
            raise ValueError(
                "Number of neighbors too large for NearestNeighborFinder! Please select KDTree!"
            ) from e

    def __getitem__(self, item):
        pos = np.zeros((self.num_pts, 3), dtype=np.float32)
        for i, neigh in enumerate(self.finder.find(item)):
            pos[i] = neigh.delta
        pos /= np.linalg.norm(pos[0])
        assert (len(pos[0]) == 3) and np.isclose(np.linalg.norm(pos[0]), 1)
        pos = torch.from_numpy(pos)
        return pos.to(torch.float32)

    def __len__(self):
        return self.particle_count


class OvitoDataSet_KDTree(Dataset):
    def __init__(self, atoms, num_pts):
        import scipy.spatial as sps

        self.num_pts = num_pts
        self.tree = sps.KDTree(atoms)

    def __getitem__(self, item):
        pos = np.zeros((self.num_pts, 3), dtype=np.float32)
        poi = self.tree.data[item]
        dd, ii = self.tree.query(poi, self.num_pts + 1)
        pos[:, :] = self.tree.data[ii[1:]] - poi
        pos[:, :] /= dd[1]
        pos = torch.from_numpy(pos)
        return pos.to(torch.float32)

    def __len__(self):
        return len(self.tree.data)


def modify(
    frame: int,
    data: DataCollection,
    model_dir: str = "",
    cutoff=0.0,
    NearestNeighborFinder=True,
    KDTree=False,
    rep=3,
    buffer=0.1,
):
    start = time.perf_counter()

    if (NearestNeighborFinder + KDTree) != 1:
        raise ValueError("Select only one option of: NearestNeighborFinder and KDTree!")
    if (not isinstance(model_dir, str)) or (len(model_dir) == 0):
        raise ValueError(f"{model_dir} is not a valid directory name!")
    if not os.path.isdir(model_dir):
        raise NotADirectoryError(f"{model_dir} does not exist!")
    if KDTree and (rep % 2) == 0:
        raise ValueError("rep needs to be an odd number!")

    with open(os.path.join(model_dir, "settings.json"), "r") as f:
        settings = json.load(f)

    mapping = {int(k): v for k, v in settings["classes"].items()}
    mapping[-1] = "Other"
    print("Possible classes:")
    for k, v in mapping.items():
        print(f"{k}: {v}")

    torch.manual_seed(settings["seed"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = Restored_Model(model_dir)

    batch_size = 8 * settings["batch_size"]
    if KDTree:
        original_num_atoms = data.particles.count
        rep = np.array((rep, rep, rep), dtype=int) * data.cell.pbc
        data.apply(
            ReplicateModifier(
                num_x=rep[0], num_y=rep[1], num_z=rep[2], adjust_box=False
            )
        )
        expr = (
            f"ReducedPosition.X <= {-buffer} || ReducedPosition.X > {1+buffer}"
            f"|| ReducedPosition.Y <= {-buffer} || ReducedPosition.Y > {1+buffer}"
            f"|| ReducedPosition.Z <= {-buffer} || ReducedPosition.Z > {1+buffer}"
        )
        data.apply(ExpressionSelectionModifier(expression=expr))
        data.apply(DeleteSelectedModifier())
        dataSet = OvitoDataSet_KDTree(data.particles["Position"], settings["num_pts"])
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

    struc[score < cutoff] = -1
    data.particles_.create_property("Structure", data=struc)
    data.particles_.create_property("Score", data=score)

    if KDTree:
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
