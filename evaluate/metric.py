import torch
import numpy as np
from parameterSetting import *
""" q-error """


def QErrorMetric(est_card, card):
    if isinstance(est_card, torch.FloatTensor) or isinstance(est_card, torch.IntTensor):
        est_card = est_card.cpu().detach().numpy()
    if isinstance(est_card, torch.Tensor):
        est_card = est_card.cpu().detach().numpy()
    est_card = np.float(est_card)
    card = np.float(card)
    if card == 0 and est_card != 0:
        return est_card
    if card != 0 and est_card == 0:
        return card
    if card == 0 and est_card == 0:
        return 1.0
    return max(est_card / card, card / est_card)


""" relative-error """


def relativeErrorMetric(est, real):
    if isinstance(est, torch.FloatTensor) or isinstance(est, torch.IntTensor):
        est = est.cpu().detach().numpy()
    if isinstance(est, torch.Tensor):
        est = est.cpu().detach().numpy()
    est = np.array(est)
    est = est.astype(np.float)
    real = real.astype(np.float)
    zero_idxs = (real == 0)
    
    est = np.abs(est - real)
    
    est[zero_idxs] = 0
    real[zero_idxs] = 1
    est /= real
    return est


def BatchErrorMetrix(est_list, oracle_list):
    ret = np.zeros(len(est_list))
    ID = 0
    if ERROR_METRIC == 'QError':
        for est, real in zip(est_list, oracle_list):
            ret[ID] = QErrorMetric(est, real)
            ID = ID + 1
    elif ERROR_METRIC == 'relative':
        ret = relativeErrorMetric(est_list, oracle_list)
    return ret