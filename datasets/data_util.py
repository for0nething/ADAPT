import os
import numpy as np
import datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Split given data into `num_clients` clients, each client has `non_iid_degree` non-iid data
def split_data_follow_non_iid_degree(data, non_iid_degree, num_clients, rng):
    num_data = data.shape[0]
    # proportion
    num_non_iid_data = int(non_iid_degree * num_data)
    num_iid_data = num_data - num_non_iid_data

    # split
    non_iid_data = data[:num_non_iid_data, :]
    iid_data = data[num_non_iid_data:, :]

    rng.shuffle(iid_data)

    # num of each client
    client_non_iid_num = int(num_non_iid_data / num_clients)
    client_iid_num = int(num_iid_data / num_clients)

    # Compute the data for each client
    client_data = []
    data_ids = []
    for i in range(num_clients - 1):
        non_iid_part = non_iid_data[client_non_iid_num * i: client_non_iid_num * (i + 1), :]
        iid_part = iid_data[client_iid_num * i: client_iid_num * (i + 1), :]
        this_client_data = np.vstack((non_iid_part, iid_part))
        rng.shuffle(this_client_data)
        client_data.append(this_client_data)

        data_ids.append(np.ones(this_client_data.shape[0]) * i)
    # Handle the last client separately
    non_iid_part = non_iid_data[(num_clients - 1) * client_non_iid_num:, :]
    iid_part = iid_data[(num_clients - 1) * client_iid_num:, :]
    this_client_data = np.vstack((non_iid_part, iid_part))
    rng.shuffle(this_client_data)
    client_data.append(this_client_data)
    data_ids.append(np.ones(this_client_data.shape[0]) * (num_clients - 1))
    return client_data, data_ids


def analyze_non_iid_ness(client_data, data_ids, num_data):
    # Concatenate the client data together
    X = np.vstack(client_data)
    y = np.concatenate(data_ids).reshape(-1, 1)

    # Create a TSNE object
    tsne = TSNE(n_components=2, n_iter=250)


    # sample 10000 data points from data
    idx = np.random.choice(num_data, 10000)
    X_tsne = tsne.fit_transform(X[idx], y=y[idx])

    X_tsne_data = np.vstack((X_tsne.T, y[idx].reshape(1, -1))).T

    plt.figure(figsize=(8, 8))

    # visualize X_tsne_data and y as label using T-SNE
    sns.scatterplot(x=X_tsne_data[:, 0], y=X_tsne_data[:, 1], hue=X_tsne_data[:, 2],
                    palette=sns.color_palette("hls", 2))
    plt.show()

