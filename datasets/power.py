from commonSetting import *
import os
import numpy as np
from evaluate.dataUtils import LoadTable

def power():
    df, _n, _dim = LoadTable("power")
    return df.values, _n, _dim

def powerSplit(split="train"):
    PATH = os.path.join(PROJECT_PATH, "dataFile/processed/power")
    file = os.path.join(PATH, f"{split}.npy")
    data = np.load(file)
    return data, data.shape[0], data.shape[1]
    # df, _n, _dim = LoadTable("power")
    # return df.values, _n, _dim