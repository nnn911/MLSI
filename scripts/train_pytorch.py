import datetime
import os
from sched import scheduler
from tempfile import tempdir

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR,
    ReduceLROnPlateau,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
import json
from Model import DGCNN_cls
from DataLoader import H5TrainingData
import tempfile
import shutil
import glob
import sys


class TrainingBase:
    def __init__(self):
        self.time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dname = os.path.join("training", self.time)
        os.makedirs(self.dname)
        print("Output directory", self.dname)
        self.n_samples = None

    def save_settings(self):
        with open(os.path.join(self.dname, "settings.json"), "w") as f:
            json.dump(self.settings, f)

    def save_checkpoint(self, fname=None):
        if fname:
            fname = os.path.join(self.dname, fname)
        else:
            fname = os.path.join(self.dname, f"torch_checkpoint_{self.epoch}.tar")
            assert not os.path.isdir(fname)

        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "loss": self.loss,
            },
            fname,
        )

    def save_full_model(self, fname):
        torch.save(self.model, os.path.join(self.dname, fname))

    def load_checkpoint(self):
        restore = os.path.join(self.prev_dname, "torch_checkpoint_best.tar")
        if not os.path.isfile(restore):
            fnames = glob.glob(os.path.join(self.prev_dname, "torch_checkpoint_*0.tar"))
            fnames.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            restore = fnames[-1]
        print(f"Restoring from {restore}")
        state_dict = torch.load(restore)

        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict["model_state_dict"], prefix="module."
        )

        return state_dict

    def train(self):
        log = open(os.path.join(self.dname, "training.log"), "w")
        log.write(
                " ".join(
                    (
                        "epoch",
                        "train_loss",
                        "train_acc",
                        "train_avg_acc",
                        "test_loss",
                        "test_acc",
                        "test_avg_acc",
                    )
                )
        )
        log.write("\n")
        detail_log = open(os.path.join(self.dname, "training_detail.log"), "w")
        self._train(log,detail_log)


class ContinueTraining(TrainingBase):
    def __init__(self, settings, prev_dname):
        super().__init__()
        self.prev_dname = prev_dname
        with open(os.path.join(self.prev_dname, "settings.json"), "r") as f:
            self.settings = json.load(f)
        self.settings.update(settings)
        if "n_samples" in self.settings:
            self.n_samples = self.settings["n_samples"]
        self.save_settings()

    def _train(self, log, detail_log):
        dataset = H5TrainingData(
            self.settings["num_pts"],
            self.settings["h5fname"],
            self.settings["mapfname"],
            self.settings["seed"],
            self.n_samples,
        )
        self.settings["num_outputs"] = dataset.num_outputs
        self.settings["classes"] = dataset.mapping_for_settings
        self.save_settings()

        train = int(len(dataset) * self.settings["split"])
        test = len(dataset) - train
        train_set, val_set = torch.utils.data.random_split(dataset, [train, test])
        train_loader = DataLoader(
            train_set,
            num_workers=8,
            batch_size=self.settings["batch_size"],
            shuffle=True,
            drop_last=True,
        )
        test_loader = DataLoader(
            val_set,
            num_workers=8,
            batch_size=self.settings["batch_size"],
            shuffle=True,
            drop_last=False,
        )

        checkpoint = self.load_checkpoint()

        device = torch.device("cuda")
        self.model = DGCNN_cls(self.settings, output_channels=dataset.num_outputs)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)

        self.model = nn.DataParallel(self.model)
        self.model = self.model.float()

        self.opt = optim.Adam(
            self.model.parameters(), lr=self.settings["lr"], weight_decay=1e-4
        )
        self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        if "schedule_divisor" in self.settings:
            self.scheduler = CosineAnnealingLR(
                self.opt,
                (self.settings["epochs"] + checkpoint["epoch"])
                // self.settings["schedule_divisor"],
                eta_min=1e-5,
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.opt, (self.settings["epochs"] + checkpoint["epoch"]), eta_min=1e-5
            )

        detail_log.write("dataset\titeration\t")
        for i in range(0,len(self.settings["classes"].keys())):
            detail_log.write(self.settings["classes"][i]+"\t")
        detail_log.write("\n")

        loss_func = nn.functional.cross_entropy

        best_test_acc = 0
        for self.epoch in range(
            checkpoint["epoch"] + 1, checkpoint["epoch"] + 1 + self.settings["epochs"]
        ):
            train_loss = 0.0
            count = 0.0
            self.model.train()
            train_pred = []
            train_true = []
            for data, label in tqdm(train_loader, leave=False):
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                logits = self.model(data)
                loss = loss_func(logits, label)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                preds = logits.max(dim=1)[1]
                count += batch_size
                train_loss += loss.item() * batch_size
                train_true.append(label.cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())
            self.loss = train_loss
            self.scheduler.step()

            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            train_acc = metrics.accuracy_score(train_true, train_pred)
            train_avg_per_class_acc = metrics.balanced_accuracy_score(
                train_true, train_pred
            )

            outstr = f"Train {self.epoch}, loss: {train_loss * 1.0 / count:.6f}, acc: {train_acc:.6f}, avg acc: {train_avg_per_class_acc:.6f}"

            logOutStr = [
                f"{self.epoch}",
                f"{train_loss * 1.0 / count}",
                f"{train_acc:.6f}",
                f"{train_avg_per_class_acc:.6f}",
            ]
            print(outstr)

            logDetailStr = "Train\t"+str(self.epoch)+"\t"
            
            for i in range(0,len(self.settings["classes"].keys())):
               sub_true = train_true[np.where(train_true==i)]
               sub_pred = train_pred[np.where(train_true==i)]
               sub_acc  = metrics.accuracy_score(sub_true,sub_pred)
               logDetailStr  +=  str(sub_acc)+"\t"
            logDetailStr+= "\n"

            test_loss = 0.0
            count = 0.0
            self.model.eval()
            test_pred = []
            test_true = []
            for data, label in tqdm(test_loader, leave=False):
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                logits = self.model(data)
                loss = loss_func(logits.squeeze(), label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())

            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            test_avg_per_class_acc = metrics.balanced_accuracy_score(
                test_true, test_pred
            )
            outstr = f"Test  {self.epoch}, loss: {test_loss * 1.0 / count:.6f}, acc: {test_acc:.6f}, avg acc: {test_avg_per_class_acc:.6f}"
            logOutStr += [
                f"{test_loss * 1.0 / count}",
                f"{test_acc}",
                f"{test_avg_per_class_acc}",
            ]
            print(outstr)
            log.write(" ".join(logOutStr))
            log.write("\n")
            log.flush()

            logDetailStr += "Test\t"+str(self.epoch)+"\t"

            for i in range(0,len(self.settings["classes"].keys())):
                sub_true = test_true[np.where(test_true==i)]
                sub_pred = test_pred[np.where(test_true==i)]
                sub_acc  = metrics.accuracy_score(sub_true,sub_pred)
                logDetailStr  +=  str(sub_acc)+"\t"
            
            logDetailStr+= "\n"
            detail_log.write(logDetailStr)
            detail_log.flush()
            
            print(logDetailStr)

            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                self.save_checkpoint("torch_checkpoint_best.tar")
                self.save_full_model("torch_model_best.t7")
            if self.epoch % 10 == 0:
                self.save_checkpoint()
        self.save_checkpoint()


class Train(TrainingBase):
    def __init__(self, settings) -> None:
        super().__init__()

        self.settings = settings

        if "n_samples" in settings:
            self.n_samples = settings["n_samples"]
        self.save_settings()

    def _train(self, log, detail_log):
        print(self.settings["num_pts"],
            self.settings["h5fname"],
            self.settings["mapfname"],
            self.settings["seed"],
            self.n_samples)
        dataset = H5TrainingData(
            self.settings["num_pts"],
            self.settings["h5fname"],
            self.settings["mapfname"],
            self.settings["seed"],
            self.n_samples,
        )
        self.settings["num_outputs"] = dataset.num_outputs
        self.settings["classes"] = dataset.mapping_for_settings
        self.save_settings()

        train = int(len(dataset) * self.settings["split"])
        test = len(dataset) - train
        train_set, val_set = torch.utils.data.random_split(dataset, [train, test])
        train_loader = DataLoader(
            train_set,
            num_workers=8,
            batch_size=self.settings["batch_size"],
            shuffle=True,
            drop_last=True,
        )
        test_loader = DataLoader(
            val_set,
            num_workers=8,
            batch_size=self.settings["batch_size"],
            shuffle=True,
            drop_last=False,
        )

        device = torch.device("cuda")
        self.model = DGCNN_cls(self.settings, output_channels=dataset.num_outputs)
        self.model.to(device)

        self.model = nn.DataParallel(self.model)
        self.model = self.model.float()

        self.opt = optim.Adam(
            self.model.parameters(), lr=self.settings["lr"], weight_decay=1e-4
        )

        eta_min = self.settings["eta_min"] if "eta_min" in self.settings else 1e-5
        scheduler_divisor = (
            self.settings["schedule_divisor"]
            if "schedule_divisor" in self.settings
            else 1
        )
        self.scheduler = CosineAnnealingLR(
            self.opt,
            self.settings["epochs"] // scheduler_divisor,
            eta_min=eta_min,
        )

        detail_log.write("dataset\titeration\t")
        for i in range(0,len(self.settings["classes"].keys())):
            detail_log.write(self.settings["classes"][i]+"\t")
        detail_log.write("\n")

        loss_func = nn.functional.cross_entropy

        best_test_acc = 0
        for self.epoch in range(self.settings["epochs"]):
            train_loss = 0.0
            count = 0.0
            self.model.train()
            train_pred = []
            train_true = []
            for data, label in tqdm(train_loader, leave=False):
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                logits = self.model(data)
                loss = loss_func(logits, label)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                preds = logits.max(dim=1)[1]
                count += batch_size
                train_loss += loss.item() * batch_size
                train_true.append(label.cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())
            self.loss = train_loss
            self.scheduler.step()

            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            train_acc = metrics.accuracy_score(train_true, train_pred)
            train_avg_per_class_acc = metrics.balanced_accuracy_score(
                train_true, train_pred
            )

            outstr = f"Train {self.epoch}, loss: {train_loss * 1.0 / count:.6f}, acc: {train_acc:.6f}, avg acc: {train_avg_per_class_acc:.6f}"

            logOutStr = [
                f"{self.epoch}",
                f"{train_loss * 1.0 / count}",
                f"{train_acc:.6f}",
                f"{train_avg_per_class_acc:.6f}",
            ]
            print(outstr)

            logDetailStr = "Train\t"+str(self.epoch)+"\t"
            
            for i in range(0,len(self.settings["classes"].keys())):
               sub_true = train_true[np.where(train_true==i)]
               sub_pred = train_pred[np.where(train_true==i)]
               sub_acc  = metrics.accuracy_score(sub_true,sub_pred)
               logDetailStr  +=  str(sub_acc)+"\t"
            logDetailStr+= "\n"

            test_loss = 0.0
            count = 0.0
            self.model.eval()
            test_pred = []
            test_true = []
            for data, label in test_loader:
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                logits = self.model(data)
                loss = loss_func(logits.squeeze(), label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())

            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            test_avg_per_class_acc = metrics.balanced_accuracy_score(
                test_true, test_pred
            )
            outstr = f"Test  {self.epoch}, loss: {test_loss * 1.0 / count:.6f}, acc: {test_acc:.6f}, avg acc: {test_avg_per_class_acc:.6f}"
            logOutStr += [
                f"{test_loss * 1.0 / count}",
                f"{test_acc}",
                f"{test_avg_per_class_acc}",
            ]
            print(outstr)
            log.write(" ".join(logOutStr))
            log.write("\n")
            log.flush()

            logDetailStr += "Test\t"+str(self.epoch)+"\t"

            for i in range(0,len(self.settings["classes"].keys())):
                sub_true = test_true[np.where(test_true==i)]
                sub_pred = test_pred[np.where(test_true==i)]
                sub_acc  = metrics.accuracy_score(sub_true,sub_pred)
                logDetailStr  +=  str(sub_acc)+"\t"
            
            logDetailStr+= "\n"
            detail_log.write(logDetailStr)
            detail_log.flush()
            
            print(logDetailStr)

            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                self.save_checkpoint("torch_checkpoint_best.tar")
                self.save_full_model("torch_model_best.t7")
            if self.epoch % 10 == 0:
                self.save_checkpoint()
        self.save_checkpoint()




def continue_run(prev_dname,points):
    settings = {
        "h5fname_source": "1000000_"+str(int(points))+"/numpy_data.h5",
        "mapfname": "1000000_"+str(int(points))+"/rev_mapping.json",
        "seed": 123456,
    }
    with tempfile.TemporaryDirectory() as tf:
        _, fname = os.path.split(settings["h5fname_source"])
        shutil.copy(settings["h5fname_source"], os.path.join(tf, fname))
        settings["h5fname"] = os.path.join(tf, fname)
        torch.manual_seed(settings["seed"])
        train = ContinueTraining(settings, prev_dname)
        train.train()


def initial_run(points):
    settings = {
        "h5fname_source": "1000000_"+str(int(points))+"/numpy_data.h5",
        "mapfname": "1000000_"+str(int(points))+"/rev_mapping.json",
        "num_pts": int(points),
        "seed": 123456,
        "batch_size": 128,
        "lr": 0.001,
        "eta_min": 1e-5,
        "k": 8,
        "dim9": False,
        "emb_dims": 16,
        "dropout": 0.2,
        "epochs": 250,
        "split": 0.8,
        "n_samples": int(5e5),
        "schedule_divisor": 1,
    }
    print(settings)
    with tempfile.TemporaryDirectory() as tf:
        _, fname = os.path.split(settings["h5fname_source"])
        shutil.copy(settings["h5fname_source"], os.path.join(tf, fname))
        settings["h5fname"] = os.path.join(tf, fname)
        torch.manual_seed(settings["seed"])
        train = Train(settings)
        train.train()




if __name__ == "__main__":
    # continue_run("training/20220829-094622")
    initial_run(sys.argv[1])

