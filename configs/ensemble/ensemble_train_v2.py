data = None
_base_ = ["../_base_/default_runtime.py"]
exp_dir = "./exp/your/save/path"

ensemble_info = [
        {'model_id': 0, 'config': './configs/ensemble/boost_model_v2.py',
         'ckpt': "./exp/your/trained/boosted/learner/path/model_best.pth"},
        {'model_id': 1, 'config': './configs/ensemble/boost_model_v2.py',
         'ckpt': "./exp/your/trained/boosted/learner/path/model_best.pth"},
        {'model_id': 2, 'config': './configs/ensemble/boost_model_v2.py',
         'ckpt': "./exp/your/trained/boosted/learner/path/model_best.pth"},
    ]


num_class = 2

lr = 1e-6
epoch = 1000
n_batch_per_epo = 20
