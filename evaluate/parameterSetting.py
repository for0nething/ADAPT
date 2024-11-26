from commonSetting import PROJECT_PATH
GPU_ID = 2



""" dataset name"""
dataset_name = 'power'
# dataset_name = 'BJAQ'
# dataset_name = 'tpch'
# dataset_name = 'flights'
# dataset_name = 'imdbfull'

numQuery = 2000
_siz = 100
_siz = int(_siz)

""" 【aggregate】 """
agg_type = 'count'
# agg_type = 'sum'
# agg_type = 'average'
# agg_type = 'variance'
# agg_type = 'percentile'
# agg_type = 'mode'
# agg_type = 'range'

aggCol = 4  # aggregate columns
PERCENT = 50 # Only for percentile query


""" error metric """
ERROR_METRIC = 'relative' # relative error
# ERROR_METRIC = 'QError' # relative error




LOAD = "JAX"   # load which model, JAX or nflows; currently only JAX (support for nflows is in the notebook)

# model path
# pickle_dir = "/home/jiayi/"
pickle_dir = PROJECT_PATH









REUSE_FROM_FILE = False
REUSE_FILE_PATH = PROJECT_PATH + 'reuseMaps/'


# do not need to change
query_type = 'conf'
group_type = 'plain'

f_batch_time = 0

""" dataset specific parameters """
if dataset_name == 'power':
    """ network parameters """
    #     hidden_features = 108
    #     num_flow_steps = 6
    hidden_features = 256
    num_flow_steps = 10

    flow_id = 1

    features = 6

    """ query settings"""
    query_seed = 45
    #     query_seed = 1234
    """ detailed network parameters"""
    anneal_learning_rate = True
    base_transform_type = 'rq-coupling'

    dropout_probability = 0
    grad_norm_clip_value = 5.
    linear_transform_type = 'lu'

    num_bins = 8
    num_training_steps = 400000
    num_transform_blocks = 2
    seed = 1638128
    tail_bound = 3
    use_batch_norm = False

    permutation_list = [[1, 0, 2, 5, 4, 3],
                        [3, 2, 1, 5, 0, 4],
                        [2, 5, 3, 1, 4, 0],
                        [2, 4, 3, 5, 1, 0],
                        [5, 3, 1, 0, 2, 4],
                        [0, 1, 5, 4, 3, 2],
                        [3, 4, 1, 2, 5, 0],
                        [2, 3, 4, 1, 0, 5],
                        [2, 5, 3, 0, 1, 4],
                        [5, 2, 4, 3, 0, 1]]

if dataset_name == 'BJAQ':
    """ network parameters """
    hidden_features = 256
    num_flow_steps = 10

    flow_id = 1

    features = 5

    """ query settings"""
    query_seed = 45

    """ detailed network parameters"""
    anneal_learning_rate = True
    base_transform_type = 'rq-coupling'

    dropout_probability = 0
    grad_norm_clip_value = 5.
    linear_transform_type = 'lu'

    num_bins = 8
    num_training_steps = 400000
    num_transform_blocks = 2
    seed = 1638128
    tail_bound = 3
    use_batch_norm = False

    permutation_list = [
        [1, 0, 4, 3, 2],
        [2, 3, 1, 0, 4],
        [3, 1, 2, 4, 0],
        [4, 3, 2, 1, 0],
        [3, 1, 0, 2, 4],
        [0, 1, 2, 4, 3],
        [2, 3, 4, 1, 0],
        [3, 2, 4, 1, 0],
        [2, 3, 0, 1, 4],
        [2, 4, 3, 0, 1]
    ]


if dataset_name == 'flights':
    dataset_name = 'flights'

    #     hidden_features = 128
    #     num_flow_steps = 4

    hidden_features = 256
    num_flow_steps = 10


    flow_id = 1

    #     features = 12
    features = 8

    """ query settings """
    query_seed = 45

    """ detailed network parameters """
    anneal_learning_rate = True
    base_transform_type = 'rq-coupling'

    dropout_probability = 0
    grad_norm_clip_value = 5.
    linear_transform_type = 'lu'

    num_bins = 8
    num_training_steps = 400000
    num_transform_blocks = 2
    seed = 1638128
    tail_bound = 3
    use_batch_norm = False

    permutation_list = [[4, 1, 6, 0, 2, 5, 3, 7],
                        [0, 4, 3, 1, 2, 5, 7, 6],
                        [3, 0, 4, 5, 2, 1, 7, 6],
                        [2, 0, 6, 4, 3, 7, 5, 1],
                        [1, 7, 5, 0, 4, 6, 3, 2],
                        [5, 4, 1, 7, 3, 6, 2, 0],
                        [3, 4, 5, 2, 6, 1, 7, 0],
                        [0, 5, 4, 1, 7, 6, 2, 3],
                        [0, 6, 2, 4, 3, 5, 7, 1],
                        [0, 2, 7, 4, 1, 3, 6, 5]]


if dataset_name == 'imdbfull':
    dataset_name = 'imdbfull'


    hidden_features = 256
    num_flow_steps = 10

    flow_id = 1

    features = 7

    """ query settings """
    query_seed = 45

    """ detailed network parameters """
    anneal_learning_rate = True
    base_transform_type = 'rq-coupling'

    dropout_probability = 0
    grad_norm_clip_value = 5.
    linear_transform_type = 'lu'

    num_bins = 8
    num_training_steps = 400000
    num_transform_blocks = 2
    seed = 1638128
    tail_bound = 3
    use_batch_norm = False

    permutation_list = [[4, 1, 6, 0, 2, 5, 3],
                        [0, 4, 3, 1, 2, 5, 6],
                        [3, 0, 4, 5, 2, 1, 6],
                        [2, 0, 6, 4, 5, 3, 1],
                        [1, 5, 0, 4, 3, 6, 2],
                        [5, 4, 1, 3, 6, 2, 0],
                        [4, 5, 2, 3, 6, 1, 0],
                        [0, 5, 4, 1, 6, 3, 2],
                        [3, 0, 6, 2, 4, 5, 1],
                        [0, 2, 3, 4, 1, 6, 5]]
