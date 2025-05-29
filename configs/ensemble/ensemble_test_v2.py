data = None
_base_ = ["../_base_/default_runtime.py"]
exp_dir = "./exp/ensemble_test"
ckpt_path = "./exp/ensemble_train/ens_20241229-21-24/ckpt/model_best.pth"

ensemble_info = [
        {'model_id': 0, 'config': './configs/ensemble/boost_model_v2.py', 'ckpt': None},
        {'model_id': 1, 'config': './configs/ensemble/boost_model_v2.py', 'ckpt': None},
        {'model_id': 2, 'config': './configs/ensemble/boost_model_v2.py', 'ckpt': None},
    ]


num_class = 2

