import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import Dataset
from ovito.data import NearestNeighborFinder
import numpy as np
import scipy.spatial as sps


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
            if dim9 is False:
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
