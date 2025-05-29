import ipdb
import torch

from utils import comm
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler

from functools import partial
from dataset import point_collate_fn
from tensorboardX import SummaryWriter
from builder import collate_fn, build_dataset, worker_init_fn, create_ddp_model

from models import build_model
from collections.abc import Mapping, Sequence

class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = self.build_model()
        self.train_loader = self.build_train_loader()
        self.val_loader = self.build_val_loader()
        self.test_dataset = self.build_test_dataset()
        # self.debug_train_loader()
        # self.debug_val_loader()

        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()

        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        
        self.comm_info = dict()
    
    def debug_train_loader(self):
        train_iter = iter(self.train_loader)
        data = next(train_iter)
        print(data['coord'].shape, data['segment'].shape)

    def debug_val_loader(self):
        val_iter = iter(self.val_loader)
        data = next(val_iter)
        print(data['coord'].shape, data['segment'].shape)

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_test_dataset(self):
        return build_dataset(self.cfg.data.test)

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model
    
    def build_scaler(self):
        scaler = torch.cuda.amp.GradScaler() if self.cfg.enable_amp else None
        return scaler

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def train(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            if comm.get_world_size() > 1:
                self.train_loader.sampler.set_epoch(self.epoch)
            self.model.train()
            self.data_iterator = enumerate(self.train_loader)
            for (self.comm_info["iter"], self.comm_info["input_dict"]) in self.data_iterator:
                self.run_step()
                loss = self.comm_info['model_output_dict']['loss'].item()
                print("### %s - Epo: %03d|%03d - Iter: %03d - Loss: %.5f" % 
                        (self.cfg['model_name'], self.epoch, self.max_epoch, self.comm_info["iter"], loss))

    def run_step(self):
        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
            print(input_dict['coord'].shape)
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict
