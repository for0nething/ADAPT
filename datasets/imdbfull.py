# Suppose we use all the join results of `movie_info_idx`, `movie_companies`, `title`, `company_name`
#
# mix.movie_id = mc.movie_id
# mc.company_id = cn.id
# mc.movie_id = title.id
import pandas as pd

# there should be 6 columns in the results:
# mix: id, info_type_id
# mc: company_type_id
# title: kind_id, production_year
# cn: country_code
# (4073078, 6)


import os
import numpy as np
from commonSetting import *

def imdbfullSplit(split="train"):
    PATH = os.path.join(PROJECT_PATH, "dataFile/processed")
    file = os.path.join(PATH, f"imdbfull_processed.npy")
    data = np.load(file)

    # np.random.shuffle(data)
    # tmp_n = int(0.1 * data.shape[0])
    # data = data[:tmp_n, :]

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

def IMDBFullPrepare():

    path = os.path.join(PROJECT_PATH, "dataFile/raw/imdb")
    save_path = os.path.join(PROJECT_PATH, "dataFile/raw/imdb")


    ##################  Load Data  ##################
    cn_name = "company_name"
    mc_name = "movie_companies"
    title_name = "title"
    mix_name = "movie_info_idx"

    cn = pd.read_csv(os.path.join(path, f"{cn_name}.csv"))
    mc = pd.read_csv(os.path.join(path, f"{mc_name}.csv"))
    title = pd.read_csv(os.path.join(path, f"{title_name}.csv"))
    mix = pd.read_csv(os.path.join(path, f"{mix_name}.csv"))


    ##################  Select used columns  ##################
    cn = cn[['id', 'country_code']]
    mc = mc[['movie_id', 'company_id', 'company_type_id']]
    title = title[['id', 'kind_id', 'production_year']]
    mix = mix[['movie_id', 'info_type_id']]

    ##################  Drop rows with missing values  ##################
    cn = cn.dropna()
    mc = mc.dropna()
    title = title.dropna()
    mix = mix.dropna()


    ##################  Encode  ##################
    #  Only factorize non-join columns!
    cn['country_code'] = pd.factorize(cn['country_code'])[0]
    mc['company_type_id'] = pd.factorize(mc['company_type_id'])[0]
    mix['info_type_id'] = pd.factorize(mix['info_type_id'])[0]

    # Note: title should be encoded into integers except production_year
    title['id'] = pd.factorize(title['id'])[0]
    title['kind_id'] = pd.factorize(title['kind_id'])[0]


    # join the tables
    mc_title = mc.merge(title, left_on='movie_id', right_on='id', how='inner')
    # drop the column 'id' of mc_title
    mc_title = mc_title.drop(columns=['id'])
    mc_title_cn = mc_title.merge(cn, left_on='company_id', right_on='id', how='inner')
    # drop the column 'id' of mc_title
    mc_title_cn = mc_title_cn.drop(columns=['id'])
    mc_title_cn_mix = mc_title_cn.merge(mix, left_on='movie_id', right_on='movie_id', how='inner')


    ##################  Save  ##################
    print(mc_title_cn_mix.shape)
    # ['movie_id', 'company_id', 'company_type_id', 'kind_id',
    #        'production_year', 'country_code', 'info_type_id']
    print(mc_title_cn_mix.columns)

    mc_title_cn_mix.to_csv(os.path.join(save_path, "imdbfull_encoded.csv"), index=False)
    np.save(os.path.join(save_path, "imdbfull_encoded.npy"), mc_title_cn_mix.to_numpy())

    # test reload
    data = np.load(os.path.join(save_path, "imdbfull_encoded.npy"))
    print(data.shape)

def IMDBFullDequantize():
    
    PATH = os.path.join(PROJECT_PATH, "dataFile/raw/imdb")

    file = os.path.join(PATH, f"imdbfull_encoded.npy")
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

    # mu
    # [1663234.9, 27729.5, 0.91545355, 2.1881678, 1987.9879, 9.050387, 1.504228]
    # s
    # [663584.44, 44231.875, 0.5710898, 1.3034894, 29.577124, 14.158293, 0.8702347]

    PATH = os.path.join(PROJECT_PATH, "dataFile/processed")
    np.save(os.path.join(PATH, f"imdbfull_processed.npy"), data)

    return data, data.shape[0], data.shape[1]


# IMDBFullPrepare()
# IMDBFullDequantize()
