
import os
import sys
import argparse
import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel


import pointcept.utils.comm as comm
from pointcept.utils.env import get_random_seed, set_seed
from pointcept.utils.config import Config, DictAction


def create_ddp_model(model, *, fp16_compression=False, **kwargs):

    if comm.get_world_size() == 1:
        return model
    # kwargs['find_unused_parameters'] = True
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
        if "output_device" not in kwargs:
            kwargs["output_device"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


def worker_init_fn(worker_id, num_workers, rank, seed):

    worker_seed = num_workers * rank + worker_id + seed
    set_seed(worker_seed)


def default_argument_parser(epilog=None):

    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
    Examples:
    Run on single machine:
        $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml
    Change some config options:
        $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001
    Run on multiple machines:
        (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
        (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--num-machines", type=int, default=1, help="total number of machines"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )

    parser.add_argument(
        "--dist-url",
        # default="tcp://127.0.0.1:{}".format(port),
        default="auto",
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "--options", nargs="+", action=DictAction, help="custom options"
    )
    return parser


def default_config_parser(file_path, options):

    if os.path.isfile(file_path):
        cfg = Config.fromfile(file_path)
    else:
        sep = file_path.find("-")
        cfg = Config.fromfile(os.path.join(file_path[:sep], file_path[sep + 1 :]))

    if options is not None:
        cfg.merge_from_dict(options)

    if cfg.seed is None:
        cfg.seed = get_random_seed()

    cfg.data.train.loop = cfg.epoch // cfg.eval_epoch

    os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)

    if not cfg.resume:
        cfg.dump(os.path.join(cfg.save_path, "config.py"))
    return cfg


def default_setup(cfg):
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
