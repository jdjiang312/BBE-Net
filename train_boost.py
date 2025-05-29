import json
import time
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import logging
import datetime

import os
import os.path as osp

from utils import comm
from train import Trainer
from tqdm import tqdm, trange
from dataset import build_dataset
from boost_model import GradientBoostModel
from torch.utils.data import DataLoader, TensorDataset
from builder import  default_setup, default_config_parser

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-c', '--config', type=str, default="./configs/ensemble/boost_model.py")
    args = parse.parse_args()
    return args

def load_cfg(config_file, ens_cfg):
    cfg = default_config_parser(config_file, None)
    cfg.num_gpu = 1
    cfg.num_worker_per_gpu = 1
    cfg.batch_size_per_gpu = ens_cfg.batch_size
    if "spunet" in config_file:
        cfg.model_name = "Spunet"
    if "st" in config_file:
        cfg.model_name = "St"
    if "ptv3" in config_file:
        cfg.model_name = "Ptv3"
    if hasattr(ens_cfg, 'data_root'):
        cfg.data['train']['data_root'] = ens_cfg.data_root
        cfg.data['test']['data_root'] = ens_cfg.data_root
        cfg.data['val']['data_root'] = ens_cfg.data_root
    if hasattr(ens_cfg, 'tra_split'):
        cfg.data['train']['split'] = ens_cfg.tra_split
    if hasattr(ens_cfg, 'tes_split'):
        cfg.data['val']['split'] = ens_cfg.tes_split
    cfg.ens_cfg = ens_cfg
    return cfg

def test_model(model, dataloader):
    data_iter = enumerate(dataloader)
    for i, input_dict in data_iter:
        input_dict = {k: v.cuda(non_blocking=True) for k, v in input_dict.items()}
        out = model(input_dict)

def build_boost_model(args, BoostModel):
    date = str(datetime.datetime.now())[:16]
    date_str = "_" + date.replace("-", "").replace(":", "").replace(" ", "-")
    ens_cfg = default_config_parser(args.config, None)
    model_list = []
    tra_loader_list = []
    val_loader_list = []
    tes_dataset_list = []

    for path in ens_cfg.config_files:
        config = load_cfg(path, ens_cfg)
        config.batch_size_val_per_gpu = 1
        trainer = Trainer(config)
        model = trainer.model
        val_loader = trainer.val_loader
        tra_loader = trainer.train_loader
        tes_dataset = trainer.test_dataset
        model_list.append(model)
        tra_loader_list.append(tra_loader)
        val_loader_list.append(val_loader)
        tes_dataset_list.append(tes_dataset)

    for i, pretrain_path in enumerate(ens_cfg.model_pretrains):
        ckpt = torch.load(pretrain_path)["state_dict"]
        model_list[i].load_state_dict(ckpt)

    boost_model = BoostModel(model_list, tra_loader_list[0], val_loader_list[0], tes_dataset_list[0])
    boost_model.ens_cfg = ens_cfg
    boost_model.exp_dir = osp.join(ens_cfg.exp_dir, osp.basename(ens_cfg.filename)[:-3])
    boost_model.exp_dir += date_str
    boost_model.logg_dir = osp.join(boost_model.exp_dir, "logg")
    os.makedirs(boost_model.logg_dir, exist_ok=True)
    boost_model.ckpt_dir = osp.join(boost_model.exp_dir, "ckpt")
    os.makedirs(boost_model.ckpt_dir, exist_ok=True)

    boost_model.n_epoch = ens_cfg.epoch
    boost_model.num_class = ens_cfg.num_class
    boost_model.n_batch_per_epo = ens_cfg.n_batch_per_epo
    boost_model.model_names = [osp.basename(k).split("-")[0] for k in ens_cfg.config_files]
    # if ens_cfg.pretrain is not None: boost_model.load_ckpt(ens_cfg.pretrain)
    return boost_model

def test_on_single_model(boost_model):
    for i in range(1,3):
        boost_model.models[i].cuda()
        boost_model.test_single_model(i)
        boost_model.models[i].cpu()
    return

if __name__ == "__main__":
    args = parse_args()
    boost_model = build_boost_model(args, GradientBoostModel)
    boost_model.print_info()
    device = torch.device("cuda:0")
    boost_model.to_device(device)

    iou_c0_list = []
    for n in range(boost_model.n_epoch):
        boost_model.train_mode()
        epo_tra_loss = boost_model.train_epoch_v2(idx=n)
        boost_model.eval_mode()
        epo_val_loss, iou_c0, iou_c1 = boost_model.evaluation_epoch_v2()

        iou_c0_list.append(iou_c0)
        boost_model.after_train_epoch(
                ckpt_name = "model_last.pth",
                logs_name = "log_last.pkl",
                logs = {"epoch": n, "train_loss": epo_tra_loss, "validation_loss": epo_val_loss, "metrics": {"IoU": [iou_c0, iou_c1]}}
            )
        if iou_c0 >= np.max(iou_c0_list):
            boost_model.after_train_epoch(
                    ckpt_name = "model_best.pth",
                    logs_name = "log_best.pkl",
                    logs = {"epoch": n, "train_loss": epo_tra_loss, "validation_loss": epo_val_loss, "metrics": {"IoU": [iou_c0, iou_c1]}}
                )
        print("Epoch Current|Total|Best %03d|%03d|%03d - Tra loss %.3f - Val loss %.3f - IoU c0 %.3f|%.3f - IoU c1 %.3f" % (n, boost_model.n_epoch, np.argmax(iou_c0_list), epo_tra_loss, epo_val_loss, iou_c0, np.max(iou_c0_list), iou_c1))

    best_ckpt_path = osp.join(boost_model.ckpt_dir, "model_best.pth")
    boost_model.load_ckpt(best_ckpt_path)
    boost_model.test_boost_model()

