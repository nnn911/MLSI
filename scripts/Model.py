import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json


class Restored_Model:
    def __init__(self, model_dir=None):
        if not model_dir:
            raise ValueError(f"{model_dir} is not a valid directory name!")
        if not os.path.isdir(model_dir):
            raise NotADirectoryError(f"{model_dir} does not exist!")
        with open(os.path.join(model_dir, "settings.json"), "r") as f:
            self.settings = json.load(f)

        self.mapping = self.settings["classes"]
        self.mapping = {int(k): v for k, v in self.mapping.items()}
        self.mapping[-1] = ["other"]

        torch.manual_seed(self.settings["seed"])
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.model = DGCNN_cls(self.settings, self.settings["num_outputs"])
        model_fname = "torch_checkpoint_best.tar"
        state_dict = torch.load(os.path.join(model_dir, model_fname))[
            "model_state_dict"
        ]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict, prefix="module."
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, *args, **kwds):
        return self.model(*args, **kwds)


class DGCNN_cls(nn.Module):
    @staticmethod
    def knn(x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)

        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
        return idx

    @staticmethod
    def get_graph_feature(x, k=20, idx=None, dim9=False):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            if dim9 == False:
                idx = DGCNN_cls.knn(x, k=k)  # (batch_size, num_points, k)
            else:
                idx = DGCNN_cls.knn(x[:, 6:], k=k)
        device = torch.device("cuda")
        # print('Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')

        idx_base = (
            torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        )

        idx = idx + idx_base

        idx = idx.view(-1)

        _, num_dims, _ = x.size()

        x = x.transpose(
            2, 1
        ).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        return feature  # (batch_size, 2*num_dims, num_points, k)

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
        x = self.get_graph_feature(
            x, k=self.settings["k"], dim9=self.settings["dim9"]
        )  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(
            x
        )  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = self.get_graph_feature(
            x1, k=self.settings["k"], dim9=self.settings["dim9"]
        )  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(
            x
        )  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = self.get_graph_feature(
            x2, k=self.settings["k"], dim9=self.settings["dim9"]
        )  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(
            x
        )  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.get_graph_feature(
            x3, k=self.settings["k"], dim9=self.settings["dim9"]
        )  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(
            x
        )  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat(
            (x1, x2, x3, x4), dim=1
        )  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(
            x
        )  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(
            batch_size, -1
        )  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(
            batch_size, -1
        )  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)  # (batch_size, emb_dims*2)

        x = F.leaky_relu(
            self.bn6(self.linear1(x)), negative_slope=0.2
        )  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(
            self.bn7(self.linear2(x)), negative_slope=0.2
        )  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)  # (batch_size, 256) -> (batch_size, output_channels)

        return x
