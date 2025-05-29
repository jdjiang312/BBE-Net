import datetime
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import logging

import os
import os.path as osp

from utils import comm
from utils.misc import intersection_and_union

from train import Trainer
from tqdm import tqdm, trange
from dataset import build_dataset
from boost_model import GradientBoostModel
from metrics import get_seg_metrics
from torch.utils.data import DataLoader, TensorDataset
from builder import  default_setup, default_config_parser
from ensemble_model import EnsembleModel

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-c', '--config', type=str, default="./configs/ensemble/ensemble_train.py")
    args = parse.parse_args()
    return args

def load_cfg(config_file, ens_cfg):
    cfg = default_config_parser(config_file, None)
    cfg.num_gpu = 1
    cfg.num_worker_per_gpu = 1
    cfg.batch_size_per_gpu = 1
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

def build_boost_core(cfg, BoostModel):
    model_list = []
    tra_loader_list = []
    val_loader_list = []
    test_dataset_list = []

    for path in cfg.config_files:
        config = load_cfg(path, cfg)
        config.batch_size_val_per_gpu = 1
        trainer = Trainer(config)
        model = trainer.model
        val_loader = trainer.val_loader
        tra_loader = trainer.train_loader
        tes_dataset = trainer.test_dataset
        model_list.append(model)
        tra_loader_list.append(tra_loader)
        val_loader_list.append(val_loader)
        test_dataset_list.append(tes_dataset)

    boost_model = BoostModel(model_list, tra_loader_list[0], val_loader_list[0], test_dataset_list[0])
    boost_model.ens_cfg = config.ens_cfg
    boost_model.exp_dir = boost_model.ens_cfg['exp_dir']
    boost_model.logg_dir = osp.join(boost_model.exp_dir, "logg")
    boost_model.ckpt_dir = osp.join(boost_model.exp_dir, "ckpt")
    os.makedirs(boost_model.logg_dir, exist_ok=True)
    os.makedirs(boost_model.ckpt_dir, exist_ok=True)

    boost_model.n_epoch = cfg.epoch
    boost_model.num_class = cfg.num_class
    boost_model.b_batch_per_epo = cfg.n_batch_per_epo
    boost_model.model_names = [osp.basename(k).split("-")[0] for k in cfg.config_files]
    return boost_model

def build_boost_model(args, BoostModel):
    models = []
    parameters = []
    ens_cfg = default_config_parser(args.config, None)

    date = str(datetime. datetime.now())[:16]
    exp_dir = osp.join(ens_cfg['exp_dir'], "ens_" + date.replace("-", "").replace(":", " ").replace(" ", "-"))
    
    for i, info in enumerate(ens_cfg.ensemble_info):
        boost_cfg = default_config_parser(info['config'], None)
        boost_cfg.exp_dir = exp_dir
        boost_model = build_boost_core(boost_cfg, BoostModel)
        print("\n")
        print("- Ensemble Model %d" % i)
        boost_model.print_info()
        try:
            boost_model.load_ckpt(info['ckpt'])
            print("Load checkpoint sucess!")
        except:
            print("Load checkpoint fail!")
        print()
        models.append(boost_model)
        parameters.append(boost_model.parameters)
    return ens_cfg, models, parameters

def save_ckpt(models, save_path):
    ens_ckpt = {}
    for i in range(len(models)):
        ens_ckpt[i] = {}
        for n, name in enumerate(models[i].model_names):
            ens_ckpt[i][name] = models[i].models[n].state_dict()
    torch.save(ens_ckpt, save_path) 
    print("Checkpoints saved at %s" % save_path)

def data_size_debug(models, key="tra"):
    print("### %s loader size debug" % key)
    for i, model in enumerate(models):
        print("Model %d debug" % i)
        model.dataloader_debug(key=key, n_step=10)

# 主流程
if __name__ == "__main__":
    args = parse_args()
    ens_cfg, models, parameters = build_boost_model(args, GradientBoostModel)
    device = torch.device("cuda:0")
    ens_model = EnsembleModel(models, parameters, device, lr=ens_cfg.lr)
    ens_model.train(n_epoch=ens_cfg.epoch)

