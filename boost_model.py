import os
import os.path as osp
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle

from tqdm import trange
from utils.misc import intersection_and_union

__all__ = ["GradientBoostModel"]


class GradientBoostModel(object):
    def __init__(
        self,
        models: List[nn.Module],
        train_loader,
        val_loader,
        test_dataset,
        lr: float = 1e-4,
        stage_lr: float = 0.1,
        n_stage_epoch: int = 5,
        num_class: int = 2,
    ):
        super().__init__()

        self.models = models
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_dataset = test_dataset
        self.num_class = num_class

        # gradient‑boost hyper‑params
        self.base_lr = lr
        self.stage_lr = stage_lr
        self.n_stage_epoch = n_stage_epoch

        self.criterion = nn.CrossEntropyLoss()

        # One independent optimiser / scheduler per learner (stage‑wise training)
        self.optimizers = [optim.Adam(m.parameters(), lr=self.base_lr) for m in self.models]
        self.schedulers = [optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
                           for opt in self.optimizers]

        # Optional: will be overwritten by outside script if needed
        self.model_names = [f"model_{i}" for i in range(len(self.models))]
        self.ckpt_dir = "./ckpts"
        self.logg_dir = "./logs"
        self.ens_cfg = {}


    def to_device(self, device: torch.device):
        for i in range(len(self.models)):
            self.models[i] = self.models[i].to(device)

    def train_mode(self, upto: Optional[int] = None):
        upto = len(self.models) - 1 if upto is None else upto
        for i in range(upto + 1):
            self.models[i].train()

    def eval_mode(self):
        for m in self.models:
            m.eval()


    def forward(
        self,
        input_dict: dict,
        *,
        upto: Optional[int] = None,
        already_on_device: bool = False,
    ) -> torch.Tensor:

        if not already_on_device:
            device = next(self.models[0].parameters()).device
            input_dict = {k: v.to(device) for k, v in input_dict.items()}
        else:
            device = next(self.models[0].parameters()).device

        if upto is None:
            upto = len(self.models) - 1

        n_point = int(input_dict["coord"].shape[0])
        boost_out = torch.zeros(n_point, self.num_class, device=device)

        for i in range(upto + 1):
            boost_out += self.stage_lr * self.models[i](input_dict)
        return boost_out


    def train_gradient_boost(self):
        device = next(self.models[0].parameters()).device
        num_stage = len(self.models)

        for k in range(num_stage):
            print(f"\n===== Train stage {k}/{num_stage - 1} =====")

            for j in range(k):
                for p in self.models[j].parameters():
                    p.requires_grad_(False)
                self.models[j].eval()

            for p in self.models[k].parameters():
                p.requires_grad_(True)
            self.models[k].train()

            opt = self.optimizers[k]
            sch = self.schedulers[k]

            for epoch in range(self.n_stage_epoch):
                epo_loss = 0.0
                tra_iter = iter(self.train_loader)

                for _ in trange(len(self.train_loader), leave=False):
                    batch = next(tra_iter)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    gt = batch["segment"]  # [N]

                    with torch.no_grad():
                        if k == 0:
                            logits_prev = torch.zeros(
                                gt.size(0), self.num_class, device=device
                            )
                        else:
                            logits_prev = self.forward(batch, upto=k - 1, already_on_device=True)

                    logits_k = self.models[k](batch)

                    logits_total = logits_prev + self.stage_lr * logits_k

                    loss = self.criterion(logits_total, gt)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    epo_loss += loss.item()

                sch.step()
                epo_loss /= len(self.train_loader)
                print(
                    f"Stage {k} | Epoch {epoch}: train CE loss = {epo_loss:.4f}")

                val_loss, iou_c0, iou_c1 = self.evaluation_epoch_v2()
                print(
                    f"              val CE {val_loss:.4f} | IoU C0 {iou_c0:.3f} C1 {iou_c1:.3f}")

            for p in self.models[k].parameters():
                p.requires_grad_(False)
            self.models[k].eval()


    def evaluation_epoch_v2(self):
        self.eval_mode()
        device = next(self.models[0].parameters()).device

        epo_loss = 0.0
        iou_c0_list: List[float] = []
        iou_c1_list: List[float] = []

        for batch in self.val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            gt = batch["segment"]

            with torch.no_grad():
                logits = self.forward(batch, already_on_device=True)
                loss = self.criterion(logits, gt)

            epo_loss += loss.item()
            pred = logits.argmax(-1).cpu().numpy()
            gt_cpu = gt.cpu().numpy()

            inter, union = intersection_and_union(pred, gt_cpu, 2, -1)
            iou_c0, iou_c1 = inter / union
            iou_c0_list.append(iou_c0)
            iou_c1_list.append(iou_c1)

        epo_loss /= len(self.val_loader)
        return epo_loss, float(np.mean(iou_c0_list)), float(np.mean(iou_c1_list))


    def test_boost_model(self):
        self.eval_mode()
        device = next(self.models[0].parameters()).device

        print("=============== Start Evaluation on Gradient Boost Model ================")
        for n in range(len(self.test_dataset)):
            print(f"- Eval data {n} ...")
            data_dict = self.test_dataset[n]
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            _ = data_dict.pop("name")  # not used here

            pred = torch.zeros((segment.size, self.num_class), device=device)
            for frag in trange(len(fragment_list), leave=False):
                input_dict = fragment_list[frag][0]
                input_dict = {k: v.to(device) for k, v in input_dict.items()}
                idx_part = input_dict["index"].cpu()
                with torch.no_grad():
                    pred_part = self.forward(input_dict, already_on_device=True)
                    pred_part = F.softmax(pred_part, dim=-1)

                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs:be], :] += pred_part[bs:be]
                        bs = be

            pred_label = pred.argmax(-1).cpu().numpy()
            inter, union = intersection_and_union(pred_label, segment, self.num_class, -1)
            iou_c0, iou_c1 = inter / union
            acc_c1 = ((pred_label == segment) & segment).sum() / segment.sum()
            acc_c0 = ((pred_label == segment) & (1 - segment)).sum() / (1 - segment).sum()
            print(
                "- Num points: %09d | Acc: C0 %.3f C1 %.3f | IoU: C0 %.3f C1 %.3f |"
                % (len(segment), acc_c0, acc_c1, iou_c0, iou_c1)
            )


    def load_ckpt(self, path):
        ckpt = torch.load(path, map_location="cpu") if isinstance(path, str) else path
        for i, name in enumerate(self.model_names):
            self.models[i].load_state_dict(ckpt[name])

    def after_train_epoch(self, ckpt_name: str, logs_name: str, logs: dict):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.logg_dir, exist_ok=True)

        model_dict = {self.model_names[i]: self.models[i].state_dict() for i in range(len(self.models))}
        ckpt_path = osp.join(self.ckpt_dir, ckpt_name)
        torch.save(model_dict, ckpt_path)
        print(f"Param saved to {ckpt_path}.")

        logs_path = osp.join(self.logg_dir, logs_name)
        with open(logs_path, "wb") as pf:
            pickle.dump(logs, pf)
        print(f"Log saved to {logs_path}.")

    def print_info(self):
        print("=============== Gradient Boost Model Info ================")
        print("Model names: ", self.model_names)
        print("Train stage epoch: ", self.n_stage_epoch)
        print("Shrinkage η: ", self.stage_lr)
        print("Base LR: ", self.base_lr)
        print("CKPT dir: ", self.ckpt_dir)
        print("Log dir: ", self.logg_dir)
        print("==========================================================")
