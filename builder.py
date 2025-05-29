
import os
import os.path as osp
import sys
import torch

import multiprocessing as mp
from utils import comm
from utils.registry import Registry
from utils.config import Config, DictAction
from collections.abc import Sequence, Mapping
from utils.env import get_random_seed, set_seed
from torch.nn.parallel import DistributedDataParallel

from boost_model import GradientBoostModel

MODELS = Registry("models")
DATASETS = Registry("datasets")

MODELS.register_module()(GradientBoostModel)


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """Wrap ``model`` with :class:`torch.nn.parallel.DistributedDataParallel` if
    world_size > 1.  No behavioural change – just moved below the new import.
    """
    if comm.get_world_size() == 1:
        return model
    # kwargs['find_unused_parameters'] = True  # uncomment if you diagnose DDP issues
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
        if "output_device" not in kwargs:
            kwargs["output_device"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


def collate_fn(batch):
    """Collate function for point clouds – unchanged."""
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)


def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    set_seed(worker_seed)


def default_config_parser(file_path, options):
    # (identical to original)
    if os.path.isfile(file_path):
        cfg = Config.fromfile(file_path)
    else:
        sep = file_path.find("-")
        cfg = Config.fromfile(os.path.join(file_path[:sep], file_path[sep + 1 :]))

    if options is not None:
        cfg.merge_from_dict(options)

    if cfg.seed is None:
        cfg.seed = get_random_seed()

    if cfg.data is not None:
        cfg.data.train.loop = cfg.epoch // cfg.eval_epoch

    os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)
    if not cfg.resume:
        cfg.dump(os.path.join(cfg.save_path, "config.py"))
    return cfg


def default_setup(cfg):
    # (identical to original)
    world_size = comm.get_world_size()
    cfg.num_worker = cfg.num_worker if cfg.num_worker is not None else mp.cpu_count()
    cfg.num_worker_per_gpu = cfg.num_worker // world_size
    assert cfg.batch_size % world_size == 0
    assert cfg.batch_size_val is None or cfg.batch_size_val % world_size == 0
    assert cfg.batch_size_test is None or cfg.batch_size_test % world_size == 0
    cfg.batch_size_per_gpu = cfg.batch_size // world_size
    cfg.batch_size_val_per_gpu = (
        cfg.batch_size_val // world_size if cfg.batch_size_val is not None else 1
    )
    cfg.batch_size_test_per_gpu = (
        cfg.batch_size_test // world_size if cfg.batch_size_test is not None else 1
    )
    assert cfg.epoch % cfg.eval_epoch == 0
    rank = comm.get_rank()
    seed = None if cfg.seed is None else cfg.seed * cfg.num_worker_per_gpu + rank
    set_seed(seed)
    return cfg


def build_model(cfg):
    return MODELS.build(cfg)


def build_dataset(cfg):
    return DATASETS.build(cfg)
