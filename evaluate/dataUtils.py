import os
from parameterSetting import *
from consts import *
import pandas as pd


# Load "original" (not normalized) data
def LoadTable(dataset_name):
    data_PATH = PROJECT_PATH + 'dataFile/raw'
    if dataset_name == "power":
        PATH = os.path.join(PROJECT_PATH, "dataFile/processed/power/")
        path = os.path.join(PATH, "train.npy")

    elif dataset_name == "BJAQ":
        path = os.path.join(data_PATH, '{}.npy'.format(dataset_name))
    elif dataset_name == "flights":
        path = os.path.join(data_PATH, "flights_encoded.npy")
    elif dataset_name == "imdbfull":
        path = os.path.join(data_PATH, "imdbfull_encoded.npy")


    else:
        assert False

    data = np.load(path).astype(np.float32)
    cols = COLS[dataset_name]

    print('data shape:', data.shape)
    n, dim = data.shape

    return pd.DataFrame(data, columns=cols), n, dim


def completeColumns(table, columns, operators, vals):
    """ complete columns not used in query"""
    ncols = table.dim
    cs = table.columns
    os, vs = [None] * ncols, [None] * ncols

    for c, o, v in zip(columns, operators, vals):
        idx = table.getColID(c if isinstance(c, str) else c.name)
        os[idx] = o
        vs[idx] = v

    return cs, os, vs


def FillInUnqueriedColumns(table, columns, operators, vals):
    ncols = len(table.columns)
    cs = table.columns
    os, vs = [None] * ncols, [None] * ncols

    for c, o, v in zip(columns, operators, vals):
        assert isinstance(c, str)
        idx = table.ColumnIndex(c)
        os[idx] = o
        vs[idx] = v

    return cs, os, vs


def LoadOracleCardinalities(dataset_name, querySeed=1234, isCluster=False, clusterNum=None):
    FILE_PATH = PROJECT_PATH + 'evaluate/oracle/{}_rng-{}.csv'.format(dataset_name, querySeed)
    if isCluster == True:
        assert clusterNum is not None
        FILE_PATH = PROJECT_PATH + 'evaluate/oracle/{}_cluster_{}_rng-{}.csv'.format(dataset_name,
                                                                                     clusterNum,
                                                                                     querySeed)

    ORACLE_CARD_FILES = {
        'power': FILE_PATH,
        'BJAQ': FILE_PATH
    }

    path = ORACLE_CARD_FILES.get(dataset_name, None)
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        print('Found oracle card!')
        return df.values.reshape(-1)
    print('Can not find oracle card! at')
    print(path)
    return None



