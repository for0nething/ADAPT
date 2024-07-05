from commonSetting import *
import os
import numpy as np
def BJAQSplit(split="train"):
    PATH = os.path.join(PROJECT_PATH, "dataFile/processed")

    file = os.path.join(PATH, f"BJAQ_processed.npy")
    data = np.load(file)


    # if split == "val" or split=="test":
    #     samples_size = int(0.05 * data.shape[0])
    #     idx = np.arange(data.shape[0])
    #     rng = np.random.RandomState(1234)
    #     rng.shuffle(idx)
    #     data = data[idx[:samples_size], :]

    idx = np.arange(data.shape[0])
    rng = np.random.RandomState(1234)
    rng.shuffle(idx)

    train_size = int(0.8 * data.shape[0])
    val_size = int(0.2 * data.shape[0])
    if split == "train":
        data = data[idx[:train_size], :]
    elif split == "val":
        data = data[idx[:val_size], :]
    elif split == "test":
        data = data[idx[train_size:], :]

    print(f"{split} data shape ", data.shape)

    return data, data.shape[0], data.shape[1]

"""
    Before using it, BJAQ is processed to store the dequantize and normalize data
    The following code only needs to do once
"""
def BJAQDequantize():
    PATH = os.path.join(PROJECT_PATH, "dataFile/raw")

    import numpy as np
    import os

    file = os.path.join(PATH, f"BJAQ.npy")
    data = np.load(file).astype(np.float32)

    # conpute the mean and std of each column
    rng = np.random.RandomState(1234)
    noise = rng.rand(data.shape[0], data.shape[1])
    noise[:, -1] *= 0.1

    data += noise
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    print("mu:")
    print(list(mu))
    print("s:")
    print(list(s))
    data = (data - mu) / s

    # save data as BJAQ_processed.npy
    PATH = os.path.join(PROJECT_PATH, "dataFile/processed")
    np.save(os.path.join(PATH, f"BJAQ_processed.npy"), data)

    return data, data.shape[0], data.shape[1]


