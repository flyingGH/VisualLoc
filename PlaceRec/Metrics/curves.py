import numpy as np
from typing import Union
import matplotlib.pyplot as plt


def pr_curve(ground_truth:np.ndarray, similarity: np.ndarray, ground_truth_soft: Union[None, np.ndarray] = None, n_thresh: int=100,
             matching: str = 'multi') -> tuple[np.ndarray, np.ndarray]:

    assert similarity.shape == ground_truth.shape, "S and GT must be the same shape"
    assert (similarity.ndim == 2), "S_in, GThard and GTsoft must be two-dimensional"
    ground_truth = ground_truth.astype('bool')
    similarity = similarity.copy()
    if ground_truth_soft is not None:
        similarity[ground_truth_soft & ~ground_truth] = similarity.min()
    # single-best-match or multi-match VPR
    if matching == 'single':
        GTP = np.count_nonzero(ground_truth.any(0))
        ground_truth = ground_truth[np.argmax(similarity, axis=0), np.arange(ground_truth.shape[1])]
        similarity = np.max(similarity, axis=0)
    elif matching == 'multi':
        GTP = np.count_nonzero(ground_truth)  # ground truth positive
    R = [0, ]
    P = [1, ]
    startV = similarity.max()  # start-value for threshold
    endV = similarity.min()  # end-value for treshold
    for i in np.linspace(startV, endV, n_thresh):
        B = similarity >= i  # apply thresh
        TP = np.count_nonzero(ground_truth & B)  # true positives
        FP = np.count_nonzero((~ground_truth) & B)  # false positives
        P.append(TP / (TP + FP))  # precision
        R.append(TP / GTP)  # recall
    return np.array(P), np.array(R)



def plot_pr_curve(ground_truth: np.ndarray, similarity: np.ndarray, ground_truth_soft: Union[None, np.ndarray] = None, n_thresh: int=100,
             matching: str = 'multi', title: str = None, show=True) -> tuple[np.ndarray, np.ndarray]:

    P, R = pr_curve(ground_truth, similarity, ground_truth_soft, n_thresh=100)

    fig, ax = plt.subplots()

    ax.plot(R, P)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("PR Curve")
    plt.show()
    return ax









