import pandas as pd
import numpy as np
import os
from commonSetting import *

def flightsSplit(split="train"):
    PATH = os.path.join(PROJECT_PATH, "dataFile/processed")

    file = os.path.join(PATH, f"flights_processed.npy")
    data = np.load(file)



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

def FlightsPrepare():
    """
        Preprocess the raw flights dataset

        Dataset link:
        https://www.kaggle.com/datasets/usdot/flight-delays
    """
    PATH = os.path.join(PROJECT_PATH, "dataFile/raw")

    flights = pd.read_csv(os.path.join(PATH, "flights.csv"))

    # output the number of distinct values of each column of flights
    print(flights.nunique())
    print()

    # output the unique values of a given column
    print(flights["LATE_AIRCRAFT_DELAY"].unique())

    # output the number of rows that have missing values for each column
    print(flights.isnull().sum())


    # get all the columns without missing values
    flights = flights.dropna(axis=1)
    print(flights.columns)


    # drop the column "YEAR" of flights
    flights = flights.drop(columns=["YEAR"])

    # encode "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT" into integers
    flights["AIRLINE"] = pd.factorize(flights["AIRLINE"])[0]
    flights["ORIGIN_AIRPORT"] = pd.factorize(flights["ORIGIN_AIRPORT"])[0]
    flights["FLIGHT_NUMBER"] = pd.factorize(flights["FLIGHT_NUMBER"])[0]
    flights["DESTINATION_AIRPORT"] = pd.factorize(flights["DESTINATION_AIRPORT"])[0]


    flights = flights.drop(columns=[ 'DAY_OF_WEEK', 'DISTANCE', 'DIVERTED', 'CANCELLED'])



    print("processed shape")
    print(flights.shape)

    flights.to_csv(os.path.join(PATH, "flights_encoded.csv"), index=False)
    np.save(os.path.join(PATH, "flights_encoded.npy"), flights.to_numpy())


    # test reload
    data = np.load(os.path.join(PATH, "flights_encoded.npy"))
    print(data.shape)

def FlightsDequantize():
    PATH = os.path.join(PROJECT_PATH, "dataFile/raw")
    import numpy as np
    import os
    # file = os.path.join(PATH, f"{split}.npy")
    file = os.path.join(PATH, f"flights_encoded.npy")
    data = np.load(file).astype(np.float32)

    # conpute the mean and std of each column
    rng = np.random.RandomState(1234)
    noise = rng.rand(data.shape[0], data.shape[1])
    data += noise
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    print("mu")
    print(list(mu))
    print("s")
    print(list(s))
    data = (data - mu) / s

    PATH = os.path.join(PROJECT_PATH, "dataFile/processed")
    np.save(os.path.join(PATH, f"flights_processed.npy"), data)

    # mu
    # [7.024034, 16.204592, 4.427125, 7.18041, 2173.5925, 112.122604, 90.102135, 1330.1022, 822.85693, 1494.3087,
    #  0.50260824, 0.51521635]
    # s
    # [3.4173267, 8.78816, 2.0097048, 4.0915203, 1757.0641, 172.18051, 160.3883, 483.75174, 607.78424, 507.16498,
    #  0.29312962, 0.31388512]

    return data, data.shape[0], data.shape[1]

# FlightsPrepare()
# FlightsDequantize()
#

