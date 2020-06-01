import numpy as np
from utils import xlogy


def cross_entropy(prediction, target):
    return -xlogy(target, prediction).sum()


g = lambda x: 2 ** x - 1
d = lambda i: 1 / np.log(i + 1)
def _dcg_k(ranked_target, k=None):
    if k is None:
        k = ranked_target.shape[0]

    ranked_target = ranked_target[:k]
    return np.sum(
        g(ranked_target) * d(np.arange(1, k + 1))
    )


def _ndcg_k(ranked_target, k=None):
    dcg_value = _dcg_k(ranked_target, k)

    ideal_dcg = _dcg_k(
        np.sort(ranked_target)[::-1],
        k
    )

    if ideal_dcg == 0:
        return 1
    return dcg_value / ideal_dcg


def dcg_k(ranked_target_list, k=None):
    scores = []
    for ranked_target in ranked_target_list:

        scores.append(
            _dcg_k(ranked_target, k)
        )

    return np.mean(scores)


def ndcg_k(ranked_target_list, k=None):
    scores = []
    for ranked_target in ranked_target_list:

        scores.append(
            _ndcg_k(ranked_target, k)
        )

    return np.mean(scores)


def _precision_at_k(ranked_target):
    return np.cumsum(ranked_target / 2) / np.arange(1, ranked_target.shape[0] + 1)


def _aprecision_at_k(ranked_target, k=None):
    if k is None:
        k = ranked_target.shape[0]

    precisions = _precision_at_k(ranked_target)[:k]
    ranked_target = ranked_target[:k]

    result = ranked_target * 1. / ranked_target.sum()

    return np.sum(result * precisions)


def map_k(ranked_target_list, k=None):
    scores = []
    for ranked_target in ranked_target_list:

        scores.append(
            _aprecision_at_k(ranked_target, k)
        )

    return np.mean(scores)
