import numpy as np
import random

def generateQuery(DW, rng):
    """ generate a query """

    num_filters = rng.randint(DW.minFilter, DW.maxFilter)

    loc = rng.randint(0, DW.cardinality)
    tuple0 = DW.data.iloc[loc]
    tuple0 = tuple0.values
    loc = rng.randint(0, DW.cardinality)
    tuple1 = DW.data.iloc[loc]
    tuple1 = tuple1.values

    idxs = rng.choice(len(DW.columns), replace=False, size=num_filters)
    cols = np.take(DW.columns, idxs)

    ops = rng.choice(['<=', '>='], size=num_filters)
    ops_all_eqs = ['='] * num_filters


    sensible_to_do_range = DW.getCateColumns(cols)
    ops = np.where(sensible_to_do_range, ops, ops_all_eqs)
    vals = tuple0[idxs]
    vals = list(vals)

    if DW.dataset_name == 'BJAQ':
        ops = ['in'] * len(vals)
        print("Len ops for BJAQ is ", len(vals))

    tuple0 = tuple0[idxs]
    tuple1 = tuple1[idxs]
    for i, op in enumerate(ops):
        if op == 'in':
            vals[i] = ([tuple0[i], tuple1[i]] if tuple0[i]<=tuple1[i] else [tuple1[i], tuple0[i]])

    return cols, ops, vals

def generateNQuery(DW, n, rng):
    """ generate N queries """
    ret = []
    for i in range(n):
        ret.append(generateQuery(DW, rng))
    return ret

