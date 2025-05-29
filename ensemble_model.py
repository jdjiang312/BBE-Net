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

def data_size_debug(models, key="tra"):
    print("### %s loader size debug" % key)
    for i, model in enumerate(models):
        print("Model %d debug" % i)
        model.dataloader_debug(key=key, n_step=10)

class EnsembleModel(object):
    def __init__(self, models, parameters, device, lr=1e-6):
        self.ckpt_dir = models[0].ckpt_dir
        self.val_loader = models[0].val_loader
        self.tra_loader = models[0].train_loader
        self.test_dataset = models[0].test_dataset

        self.models = models
        self.device = device
        self.to_device()
        self.N_model = len(models)

        self.parameters = []
        for param in parameters: self.parameters += param

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters, lr=lr)
    
    def to_device(self):
        for i in range(len(self.models)):
            self.models[i].to_device(self.device)
        return
    
    def load_ckpt(self, path):
        ens_ckpt = torch.load(path, map_location='cpu')
        for i in range(self.N_model): 
            try:
                self.models[i].load_ckpt(ens_ckpt[i])
                print("- Boost model %d load checkpoint success!" % i)
            except:
                print("- Boost model %d load checkpoint failed!" % i)

    def save_ckpt(self, save_path):
        ens_ckpt = {}
        for i in range(self.N_model):
            ens_ckpt[i] = {}
            for n, name in enumerate(self.models[i].model_names):
                ens_ckpt[i][name] = self.models[i].models[n].state_dict()
        torch.save(ens_ckpt, save_path) 
        print("Checkpoints saved at %s" % save_path)

    def test(self, path):
        self.load_ckpt(path)
        for i in range(self.N_model): self.models[i].eval_mode()
        print("========== Start Test on Ensemble Model ==========")
        for n in range(self.N_model):
            print("- Eval data %d ..." % n)
            data_dict = self.test_dataset[n]
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            n_frag = len(fragment_list)
            pred = torch.zeros((segment.size, 2))

            for i in trange(n_frag):
                fragment_batch_size = 1
                s_i = i*fragment_batch_size
                e_i = min((i+1)*fragment_batch_size, len(fragment_list))
                input_dict = fragment_list[s_i:e_i][0]
                idx_part = input_dict["index"].cpu()
                with torch.no_grad():
                    pred_part = self.forward(input_dict).cpu()
                    pred_part = F.softmax(pred_part, -1)
                    torch.cuda.empty_cache()
                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs:be], :] += pred_part[bs:be]
                        bs = be
                input_dict = {k: v.cpu() for k,v in input_dict.items()}
            pred = pred.max(1)[1].data.cpu().numpy()
            out = intersection_and_union(pred, segment, 2, -1)
            iou_c0, iou_c1 = (out[0] / out[1])
            acc_c1 = ((pred == segment) & segment).sum() / segment.sum()
            acc_c0 = ((pred == segment) & (1-segment)).sum() / (1-segment).sum()
            print("- Num points: %09d | Acc: Class0 %.3f Class1 %.3f | IoU: Class0 %.3f Class1 %.3f |" % (len(segment), acc_c0, acc_c1, iou_c0, iou_c1))

    def train(self, n_epoch=10):
        self.save_ckpt(osp.join(self.models[0].ckpt_dir, "model_last.pth"))
        
        mIoU_c0_list = []
        for n in range(n_epoch):
            epo_tra_loss = 0
            n_step = len(self.tra_loader)
            for i in range(self.N_model): self.models[i].train_mode()

            tra_iter = iter(self.tra_loader)
            for n in trange(n_step):
                input_dict = next(tra_iter)
                gt = input_dict["segment"]
                self.optimizer.zero_grad()
                ens_out = self.forward(input_dict)
                gt = gt.to(ens_out.device)
                loss = self.criterion(ens_out, gt)
                loss.backward()
                self.optimizer.step()
                epo_tra_loss += loss.item()
            epo_tra_loss /= n_step

            epo_val_loss = 0
            iou_c0_list, iou_c1_list = [], []
            for i in range(self.N_model): self.models[i].eval_mode()
            for i, input_dict in enumerate(self.val_loader):
                gt = input_dict['segment']
                gt = gt.to(ens_out.device)
                with torch.no_grad():
                    ens_out = self.forward(input_dict)
                    loss = self.criterion(ens_out, gt)
                ens_pred = ens_out.max(1)[1].cpu().numpy()
                epo_val_loss += loss.item()
                out = intersection_and_union(ens_pred, gt.cpu().numpy(), 2, -1)
                iou_c0, iou_c1 = (out[0] / out[1])
                iou_c0_list.append(iou_c0)
                iou_c1_list.append(iou_c1)
            epo_val_loss /= len(self.val_loader)
            mean_iou_c0 = np.mean(iou_c0_list)
            mean_iou_c1 = np.mean(iou_c1_list)
            mIoU_c0_list.append(mean_iou_c0)
            
            self.save_ckpt(osp.join(self.models[0].ckpt_dir, "model_last.pth"))
            if mean_iou_c0 >= np.max(mIoU_c0_list): self.save_ckpt(osp.join(self.models[0].ckpt_dir, "model_best.pth"))
            print("Epoch %03d|%03d|%03d - mIoU class0 %.3f|%.3f" % (n, n_epoch, np.argmax(mIoU_c0_list), mean_iou_c0, np.max(mIoU_c0_list)))
            
    def forward(self, input_dict):
        input_dict = {k: v.cpu() for k, v in input_dict.items()}
        n_points = int(input_dict['coord'].shape[0])
        out = torch.zeros((n_points,2)).to(self.device)
        for i, boost_model in enumerate(self.models):
            boost_out = boost_model.forward(input_dict)
            out += boost_out
        return out
    
