data = None
_base_ = ["../_base_/default_runtime.py"]

exp_dir = "./exp/your/save/path"
data_root = "/your/dataset/path"
config_files = [
        "./configs/s3dis/spunet-ensemble.py",
        "./configs/s3dis/st-ensemble.py",
        "./configs/s3dis/ptv3-ensemble.py"
    ]
model_pretrains = [
        "./ckpt/SpUnet/spunet_model_best.pth",
        "./ckpt/ST/st_model_best.pth",
        "./ckpt/PTv3/ptv3_model_best.pth",
    ]

pretrain = None

num_class = 2
tra_split = ("Area_1", "Area_2", "Area_3", "Area_4", "Area_6")
tes_split = ("Area_5")

epoch = 1000
batch_size = 1
n_batch_per_epo = 20
