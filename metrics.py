import numpy as np

def get_seg_metrics(pred, GT, num_class=2):
    res = {}
    for cls in range(num_class):
        GT_cls = GT == cls
        inter = (pred == cls) & GT_cls
        union = (pred == cls) | GT_cls
        iou = inter.sum() / (union.sum() + 1e-9)
        acc = inter.sum() / (GT_cls.sum() + 1e-9)
        res[cls] = {"acc": acc, "iou": iou, "num": GT_cls.sum()}
    return res
