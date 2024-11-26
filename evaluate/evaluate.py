from dataUtils import LoadTable
from parameterSetting import *
from parameterSetting import _siz
from dataWrapper import *
from genQuery import *
from metric import *
from modelConstruct import *
from hiddenUnit import *
from integrand import *
from reuse import *
from estimate import *
import numpy as np
import os, sys
import torch
from torchquad import  enable_cuda, VEGAS
import time
from nflows import transforms
from nflows import distributions
from nflows import utils
from nflows import flows
import pickle
from commonSetting import PROJECT_PATH


os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(GPU_ID)
assert torch.cuda.is_available()
device = torch.device('cuda')

sys.path.append(os.path.abspath(PROJECT_PATH + 'utils/'))
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
DEVICENAME = torch.cuda.get_device_name(0)
print('DEVICE NAME\n', DEVICENAME)
from torchquad import set_up_backend
set_up_backend("torch", data_type="float32")




"""Load trained model"""
# model structure
distribution = distributions.StandardNormal((features,))
transform = create_transform(features)
flow = flows.Flow(transform, distribution).to(device)

print("see flow")
print(flow)

# model params
with open(os.path.join(pickle_dir, "model_dict.pickle"), "rb") as f:
    model_dict = pickle.load(f)
print(model_dict)
print(len(transform._transforms))

transform = assign_transform_params(model_dict, transform)

# get the correct flow
flow = flows.Flow(transform, distribution).to(device)
flow.cuda()
flow.eval()


n_params = utils.get_num_parameters(flow)
print('There are {} trainable parameters in this model.'.format(n_params))
print('Parameters total size is {} MB'.format(n_params * 4 / 1024 / 1024))





""" Build Datawrapper """
data, n, dim = LoadTable(dataset_name)
DW = DataWrapper(data, dataset_name)
rng = np.random.RandomState(query_seed)



""" Generate Queries"""
rng = np.random.RandomState(query_seed)
IfLoadOracle = False
queries = generateNQuery(DW, numQuery, rng)



""" Compute Oracle """
""" Load oracle_cards """
oracle_cards_arr = None

if IfLoadOracle == True:
    oracle_cards_arr = LoadOracleCardinalities(dataset_name, query_seed)
if oracle_cards_arr is None:
    oracle_cards_arr = DW.getAllOracle(queries, agg_type=agg_type)

# save the oracle of all methods
# np.save("oracle_selectivity_{}.npy".format(dataset_name), oracle_cards_arr/DW.n)
# exit()

legal_lists = DW.getLegalRangeNQuery(queries)
legal_tensors = torch.Tensor(legal_lists).to('cuda')





def getClustersPlain(query_tensors, oracles_cards, group_siz):
    n, dim, _ = query_tensors.shape
    # hyperparameter
    n_group = int(n / group_siz)

    # 【clusters】
    # [groupN, dim, 2]

    # get MIN
    tmp = query_tensors[:, :, 0].view(-1, group_siz, dim)
    MIN = torch.min(tmp, dim=1)[0]
    # get MAX
    tmp = query_tensors[:, :, 1].view(-1, group_siz, dim)
    MAX = torch.max(tmp, dim=1)[0]
    cluster = torch.stack((MIN, MAX), dim=-1)

    # 【weights】
    _weights = torch.ones(n_group, dtype=int) * group_siz
    return cluster, _weights, query_tensors, oracles_cards


def Group(group_type, legal_tensors, oracle_cards=oracle_cards_arr, groupSize=_siz):
    legal_tensors = legal_tensors.clone().detach()
    oracle_cards = oracle_cards.copy()
    if group_type == 'plain':
        clusterQueries, clusterWeights, legal_tensors, oracle_cards = getClustersPlain(legal_tensors, oracle_cards,
                                                                                       groupSize)
    return clusterQueries, clusterWeights, legal_tensors, oracle_cards


# clusterQueries, clusterWeights, legal_tensors, oracle_cards = Group(group_type)
_, __, legal_tensors, oracle_cards = Group(group_type, legal_tensors, oracle_cards=oracle_cards_arr)
print(_.shape)

enc = DW.getNColumnEnc(queries)
print(enc.shape)


f_batch, f_batch_count, f_batch_sum, f_batch_square_sum = get_f_batch(flow, DW)


target_map, domain_starts, domain_sizes = reuse(DW, f_batch)







#### get result
alpha_list = [0]
beta_list = [0.4]


p50s = []
p95s = []
p99s = []
pmaxs = []
means = []

for alpha in alpha_list:
    for beta in beta_list:

        if agg_type in ['count', 'sum']:
            p50, p95, p99, pmax, mean, result = testHyper(40, 50000, 4, alpha, beta, ifGroup=False,  oracle_cards=oracle_cards, f_batch=f_batch,
                                                          DW=DW, legal_tensors=legal_tensors, target_map=target_map, domain_starts=domain_starts, domain_sizes=domain_sizes)

        elif agg_type == 'average':
            ################  count  ##############
            agg_type = 'count'
            f_batch = f_batch_count
            p50, p95, p99, pmax, mean, count_result = testHyper(40, 50000, 4, alpha, beta, ifGroup=False, oracle_cards=oracle_cards, f_batch=f_batch,
                                                                DW=DW, legal_tensors=legal_tensors,
                                                                target_map=target_map, domain_starts=domain_starts,
                                                                domain_sizes=domain_sizes)

            ################  sum  ##############
            agg_type = 'sum'
            f_batch = f_batch_sum

            p50, p95, p99, pmax, mean, sum_result = testHyper(40, 50000, 4, alpha, beta, ifGroup=False, oracle_cards=oracle_cards, f_batch=f_batch,
                                                              DW=DW, legal_tensors=legal_tensors, target_map=target_map, domain_starts=domain_starts, domain_sizes=domain_sizes)
            avg_result = sum_result / count_result

            oracle_list = oracle_cards.copy()
            err_list = BatchErrorMetrix(avg_result, oracle_list)

            p50 = np.percentile(err_list, 50)
            p95 = np.percentile(err_list, 95)
            p99 = np.percentile(err_list, 99)
            pmax = np.max(err_list)
            mean = np.average(err_list)
            print('Mean [{:.3f}]  Median [{:.3f}]  95th [{:.3f}]  99th [{:.3f}]  max [{:.3f}]'.format(mean, p50, p95,
                                                                                                      p99, pmax))


        elif agg_type == 'variance':
            ################  count  ##############
            agg_type = 'count'
            f_batch = f_batch_count
            p50, p95, p99, pmax, mean, count_result = testHyper(40, 50000, 4, alpha, beta, ifGroup=False, oracle_cards=oracle_cards, f_batch=f_batch,
                                                                DW=DW, legal_tensors=legal_tensors,
                                                                target_map=target_map, domain_starts=domain_starts,
                                                                domain_sizes=domain_sizes)

            ################  sum  ##############
            agg_type = 'sum'
            f_batch = f_batch_sum
            p50, p95, p99, pmax, mean, sum_result = testHyper(40, 50000, 4, alpha, beta, ifGroup=False, oracle_cards=oracle_cards, f_batch=f_batch,
                                                              DW=DW, legal_tensors=legal_tensors, target_map=target_map, domain_starts=domain_starts, domain_sizes=domain_sizes)

            ################  square sum  ##############
            agg_type = 'variance'
            f_batch = f_batch_square_sum
            p50, p95, p99, pmax, mean, square_sum_result = testHyper(40, 50000, 4, alpha, beta, ifGroup=False, oracle_cards=oracle_cards, f_batch=f_batch,
                                                                     DW=DW, legal_tensors=legal_tensors,
                                                                     target_map=target_map, domain_starts=domain_starts,
                                                                     domain_sizes=domain_sizes)

            ################  average  ##############
            avg_result = sum_result / count_result

            y2_result = square_sum_result / count_result

            var_result = y2_result - avg_result ** 2

            oracle_list = oracle_cards.copy()
            err_list = BatchErrorMetrix(var_result, oracle_list)

            p50 = np.percentile(err_list, 50)
            p95 = np.percentile(err_list, 95)
            p99 = np.percentile(err_list, 99)
            pmax = np.max(err_list)
            mean = np.average(err_list)
            print('Mean [{:.3f}]  Median [{:.3f}]  95th [{:.3f}]  99th [{:.3f}]  max [{:.3f}]'.format(mean, p50, p95,
                                                                                                      p99, pmax))

        elif agg_type == 'percentile':
            agg_type = 'count'
            f_batch = f_batch_count

            """ 【hyper param】# of binary search """
            binary_search_limit = 20

            p50, p95, p99, pmax, mean, percent_result = testHyperForPercentile(40, 50000, 4, alpha, beta,
                                                                               binary_search_limit=binary_search_limit,
                                                                               ifGroup=False,
                                                                                oracle_cards=oracle_cards, f_batch=f_batch,
                                                                                DW=DW, legal_tensors=legal_tensors,
                                                                                target_map=target_map, domain_starts=domain_starts,
                                                                                domain_sizes=domain_sizes)

            oracle_list = oracle_cards.copy()

            # for debug
            # print("see estimated results")
            # print(percent_result)
            # print("see oracle values")
            # print(oracle_list)

            err_list = BatchErrorMetrix(percent_result, oracle_list)

            p50 = np.percentile(err_list, 50)
            p95 = np.percentile(err_list, 95)
            p99 = np.percentile(err_list, 99)
            pmax = np.max(err_list)
            mean = np.average(err_list)


        elif agg_type == 'range':
            agg_type = 'count'
            f_batch = f_batch_count

            """ 【hyper param】# of binary search """
            binary_search_limit = 20

            PERCENT = 0.9999
            p50, p95, p99, pmax, mean, max_result = testHyperForPercentile(40, 50000, 4, alpha, beta,
                                                                               binary_search_limit=binary_search_limit,
                                                                               ifGroup=False,
                                                                                oracle_cards=oracle_cards, f_batch=f_batch,
                                                                                DW=DW, legal_tensors=legal_tensors,
                                                                                target_map=target_map, domain_starts=domain_starts,
                                                                                domain_sizes=domain_sizes)

            PERCENT = 0.001

            # binary search for left side for computing MIN
            p50, p95, p99, pmax, mean, min_result = testHyperForPercentile(40, 50000, 4, alpha, beta,
                                                                           binary_search_limit=binary_search_limit,
                                                                           ifGroup=False,
                                                                           oracle_cards=oracle_cards, f_batch=f_batch,
                                                                           DW=DW, legal_tensors=legal_tensors,
                                                                           target_map=target_map,
                                                                           domain_starts=domain_starts,
                                                                           domain_sizes=domain_sizes,
                                                                           binary_side='left')

            oracle_list = oracle_cards.copy()
            range_result = max_result - min_result
            # for debug
            # print("see estimated results")
            # print(" ---- min")
            # print(min_result)
            # print(" ---- max")
            # print(max_result)
            # print(" ---- range")
            # print(range_result)
            #
            # print("see oracle values")
            # print(oracle_list)

            err_list = BatchErrorMetrix(range_result, oracle_list)

            p50 = np.percentile(err_list, 50)
            p95 = np.percentile(err_list, 95)
            p99 = np.percentile(err_list, 99)
            pmax = np.max(err_list)
            mean = np.average(err_list)




        elif agg_type =='mode':
            # transform each MODE query to several COUNT queries
            COUNT_queries, COUNT_aggCol_values, MODE_mapping = transform_MODE_queries(DW, queries)
            # compute oracle cards for the COUNT queries
            COUNT_oracle_cards_arr = DW.getAllOracle(COUNT_queries, agg_type="count")


            print("see COUNT queries")
            print(COUNT_queries)
            print("see MODE mapping")
            print(MODE_mapping)

            # legal lists for integration
            COUNT_legal_lists = DW.getLegalRangeNQuery(COUNT_queries)
            nCOUNTqueries = len(COUNT_legal_lists)


            groupSize = 100 # for count queries
            if nCOUNTqueries < groupSize:
                groupSize = nCOUNTqueries

            # This may not be divided by groupSiz
            # padding with the last element in COUNT_legal_lists
            if len(COUNT_legal_lists) % groupSize != 0:
                pad = groupSize - len(COUNT_legal_lists) % groupSize
                last_element = COUNT_legal_lists[-1]
                last_oracle = COUNT_oracle_cards_arr[-1]
                # get a list with pad last_element
                for i in range(pad):
                    COUNT_legal_lists.append(last_element)
                    COUNT_oracle_cards_arr= np.append(COUNT_oracle_cards_arr, last_oracle)


            COUNT_legal_tensors = torch.Tensor(COUNT_legal_lists).to('cuda')


            _, __, COUNT_legal_tensors, COUNT_oracle_cards = Group(group_type, COUNT_legal_tensors, oracle_cards=COUNT_oracle_cards_arr, groupSize=groupSize)
            print(_.shape)
            print("total COUNT queries", COUNT_legal_tensors.shape)
            print("see COUNT oracle cards shape ", COUNT_oracle_cards.shape)

            # have to set the numQuery here!!!
            p50, p95, p99, pmax, mean, result = testHyper(40, 50000, 4, alpha, beta,
                                                          numQuery = COUNT_legal_tensors.shape[0],
                                                          ifGroup=False,  oracle_cards=COUNT_oracle_cards, f_batch=f_batch,
                                                          DW=DW, legal_tensors=COUNT_legal_tensors, target_map=target_map, domain_starts=domain_starts, domain_sizes=domain_sizes)

            print("total result size", result.shape)
            print("see COUNT aggCol values ")
            print(COUNT_aggCol_values)
            print("see COUNT results")
            print(list(result.numpy()))

            # todo: compute the estimated MODE values
            mode_result = estimate_MODE_results(result, MODE_mapping, COUNT_aggCol_values)

            oracle_list = oracle_cards.copy()

            # for debug
            print("see estimated MODE results")
            print(mode_result)
            print("see oracle values")
            print(oracle_list)

            err_list = BatchErrorMetrix(mode_result, oracle_list)

            p50 = np.percentile(err_list, 50)
            p95 = np.percentile(err_list, 95)
            p99 = np.percentile(err_list, 99)
            pmax = np.max(err_list)
            mean = np.average(err_list)


        p50s.append(p50)
        p95s.append(p95)
        p99s.append(p99s)
        pmaxs.append(pmax)
        means.append(mean)

        print("[Final Result]")
        print('Mean [{:.3f}]  Median [{:.3f}]  95th [{:.3f}]  99th [{:.3f}]  max [{:.3f}]'.format(mean, p50, p95,
                                                                                                  p99, pmax))
