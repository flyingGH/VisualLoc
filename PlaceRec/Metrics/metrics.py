import numpy as np
from sklearn.metrics import precision_recall_curve
from .curves import pr_curve
from typing import Union


def recallatk(ground_truth: np.ndarray, preds: np.ndarray, similarity: np.ndarray, ground_truth_soft: Union[None, np.ndarray] = None, k: int=1) -> float:
    assert ground_truth.ndim == 2
    assert ground_truth.shape == preds.shape == similarity.shape

    if ground_truth_soft is not None:
        similarity[ground_truth_soft & ~ground_truth] = similarity.min()
    sorted_idxs = np.argsort(-similarity)
    top_idxs = np.unravel_index(sorted_idxs, similarity.shape)
    top_preds = preds[top_idxs]
    top_gt = ground_truth[top_idxs]
    print("=======================================", np.sum(top_gt))
    TP = np.count_nonzero(top_gt & top_preds)
    GTP = np.count_nonzero(top_gt)
    if GTP == 0:
        raise Exception("Divide by zero. GTP: 0")
    return TP/GTP

def precisionatk(ground_truth: np.ndarray, preds: np.ndarray, similarity: np.ndarray, ground_truth_soft: Union[None, np.ndarray] = None, k: int=1) -> float:
    if ground_truth_soft is not None:
        similarity[ground_truth_soft & ~ground_truth] = similarity.min()
    sorted_idxs = np.argsort(-similarity)
    top_idxs = np.unravel_index(sorted_idxs, similarity.shape)
    top_preds = preds[top_idxs]
    top_gt = ground_truth[top_idxs]
    TP = np.count_nonzero(top_gt & top_preds)
    FP = np.count_nonzero((~top_gt) & top_preds)
    if TP + FP == 0:
        raise Exception("Divide by zero. TP: " + str(TP) + "  FP: " + str(FP))
    return TP/(TP + FP)


def recallatprecision(ground_truth: np.ndarray, preds: np.ndarray, similarity: np.ndarray, ground_truth_soft: Union[None, np.ndarray] = None, precision: float=1.) -> float:
    if ground_truth_soft is not None:
        similarity[ground_truth_soft & ~ground_truth] = similarity.min()
    P, R = pr_curve(ground_truth, similarity)
    R = R[P >= precision]
    R = R.max()
    return R
    

def precision(ground_truth: np.ndarray, preds: np.ndarray, similarity:np.ndarray, ground_truth_soft: Union[None, np.ndarray] = None,) -> float:
    if ground_truth_soft is not None:
        similarity[ground_truth_soft & ~ground_truth] = similarity.min()
    preds = preds.astype(bool)
    ground_truth = ground_truth.astype(bool)
    TP = np.count_nonzero(ground_truth & preds)
    FP = np.count_nonzero((~ground_truth) & preds)
    if TP + FP == 0:
        raise Exception("Divide by zero. TP: " + str(TP) + "  FP: " + str(FP))
    return TP/(TP + FP)


def recall(ground_truth: np.ndarray, preds: np.ndarray, similarity:np.ndarray, ground_truth_soft: Union[None, np.ndarray] = None,) -> float:
    if ground_truth_soft is not None:
        similarity[ground_truth_soft & ~ground_truth] = similarity.min()
    preds = preds.astype(bool)
    ground_truth = ground_truth.astype(bool)
    TP = np.count_nonzero(ground_truth & preds)
    GTP = np.count_nonzero(ground_truth)
    if GTP == 0:
        raise Exception("Divide by zero. GTP: 0")
    return TP/GTP

