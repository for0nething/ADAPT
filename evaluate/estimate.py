from parameterSetting import *
import torch
from torchquad import  enable_cuda, VEGAS
import pickle
import time
from torchquad import BatchMulVEGAS
from hiddenUnit import HiddenPrints
from metric import *
from tqdm import trange
from consts import OPS

def getResultMaskConf(n, N, num_iterations=3, alpha=0.5, beta=0.5,
                      f_batch=None, legal_tensors=None, target_map=None, domain_starts=None, domain_sizes=None, numQuery=numQuery):
    global f_batch_time

    """ n: batch size """
    z = BatchMulVEGAS()
    DIM = features
    #     full_integration_domain = torch.Tensor(DIM * [[0,1]])
    #     inp = full_integration_domain.clone().unsqueeze(0)

    start_id = 0
    end_id = 0

    f_batch_time = 0
    st = time.time()
    results = []
    with torch.no_grad():
        while start_id < numQuery:
            end_id = end_id + n
            if end_id > numQuery:
                end_id = numQuery

            z.setValues(f_batch,
                        dim=DIM,
                        alpha=alpha,
                        beta=beta,
                        N=N,
                        n=end_id - start_id,
                        iterations=num_iterations,
                        #                     integration_domains=full_integration_domain.unsqueeze(0),
                        integration_domains=legal_tensors[start_id:end_id],
                        rng=None,
                        seed=1234,
                        reuse_sample_points=True,
                        target_map=target_map,
                        target_domain_starts=domain_starts,
                        target_domain_sizes=domain_sizes,
                        mask_integration_domains=legal_tensors[start_id:end_id],
                        #                     mask_integration_domains=inp.repeat(end_id - start_id,1,1),
                        useMask=True,
                        integration_weights=torch.ones(end_id - start_id, dtype=int)
                        )
            start_id = start_id + n

            results.append(z.integrate())

    en = time.time()
    total_time = en - st
    return total_time, results



########### end-to-end function
def testHyper(n, N, num_iterations, alpha, beta, ifGroup=0,
              DW=None, oracle_cards=None, f_batch=None, legal_tensors=None, target_map=None, domain_starts=None, domain_sizes=None, numQuery=numQuery):
    global groupTime
    with HiddenPrints():
        #         total_time, result = getResult(n=n,

        total_time, result = getResultMaskConf(n=n,
                                               N=N,
                                               num_iterations=num_iterations,
                                               alpha=alpha,
                                               beta=beta,
                                               f_batch=f_batch,
                                               legal_tensors=legal_tensors,
                                               target_map=target_map,
                                               domain_starts=domain_starts,
                                               domain_sizes=domain_sizes,
                                               numQuery=numQuery)

        result = torch.cat(tuple(result))
        FULL_SIZE = torch.Tensor([DW.n])

        result = result * FULL_SIZE

        result = result.to('cpu')

        n_ = numQuery
        oracle_list = oracle_cards.copy()

    err_list = BatchErrorMetrix(result, oracle_list)
    with HiddenPrints():
        if ifGroup is False:
            groupTime = 0

        total_query_time = total_time + groupTime

        avg_per_query_time = 1000. * (total_query_time / n_)
        avg_group_time = 1000. * groupTime / n_
        avg_f_batch_time = 1000. * f_batch_time / n_

        avg_vegas_time = avg_per_query_time - avg_f_batch_time - avg_group_time

    print("********** total_n=[{}] batchn=[{}]  N=[{}]  nitr=[{}]  alpha=[{}]  beta=[{}] ******".format(n_, n, N,
                                                                                                        num_iterations,
                                                                                                        alpha, beta))
    print('@ Average per query          [{}] ms'.format(avg_per_query_time))

    print(' --  Average per query group [{}] ms'.format(avg_group_time))
    print(' --  Average per query NF    [{}] ms'.format(avg_f_batch_time))
    print(' --  Average per query vegas [{}] ms'.format(avg_vegas_time))
    p50 = np.percentile(err_list, 50)
    p95 = np.percentile(err_list, 95)
    p99 = np.percentile(err_list, 99)
    pmax = np.max(err_list)
    mean = np.average(err_list)
    print('Mean [{:.3f}]  Median [{:.3f}]  95th [{:.3f}]  99th [{:.3f}]  max [{:.3f}]'.format(mean, p50, p95,
                                                                                              p99, pmax))

    return p50, p95, p99, pmax, mean, result





# binary_side   the side for binary search
def testHyperForPercentile(n, N, num_iterations, alpha, beta,
                            binary_search_limit,
                           ifGroup=0,
              DW=None, oracle_cards=None, f_batch=None, legal_tensors=None, target_map=None, domain_starts=None, domain_sizes=None,
            binary_side='right'):
    global groupTime
    # 1. selectivity for the whole query domain
    # 2. binary search for the percentile value
    with HiddenPrints():
        total_time, result = getResultMaskConf(n=n,
                                               N=N,
                                               num_iterations=num_iterations,
                                               alpha=alpha,
                                               beta=beta,
                                               f_batch=f_batch,
                                               legal_tensors=legal_tensors,
                                               target_map=target_map,
                                               domain_starts=domain_starts,
                                               domain_sizes=domain_sizes)
    with torch.no_grad():
        result = torch.cat(tuple(result))
        print("the originial result shape is ", result.shape)
        target_result = result * (PERCENT / 100.)

        bs_l = legal_tensors[:, aggCol, 0].clone()
        bs_r = legal_tensors[:, aggCol, 1].clone()

    # binary search for the percentile result
    pbar = trange(binary_search_limit, desc="Binary Search Rounds", miniters=1)
    for i in pbar:
        with torch.no_grad():
            if binary_side == 'right':
                legal_tensors[:, aggCol, 1] = (bs_l + bs_r)/2
            elif binary_side =='left':
                legal_tensors[:, aggCol, 0] = (bs_l + bs_r)/2
            else:
                assert False
        # print("For debug!")
        # print(legal_tensors[:, aggCol, 1])
        with HiddenPrints():
            this_time, result = getResultMaskConf(n=n,
                                                   N=N,
                                                   num_iterations=num_iterations,
                                                   alpha=alpha,
                                                   beta=beta,
                                                   f_batch=f_batch,
                                                   legal_tensors=legal_tensors,
                                                   target_map=target_map,
                                                   domain_starts=domain_starts,
                                                   domain_sizes=domain_sizes)
        # update in binary search
        with torch.no_grad():
            total_time += this_time
            result = torch.cat(tuple(result))
            # for debug
            # print("see result count")
            # print(result.shape)
            # print(result)
            # print("see target result count")
            # print(target_result.shape)
            # print(target_result)

            if binary_side == 'right':
                idx = torch.where(result > target_result)[0]
                bs_r[idx] = legal_tensors[idx, aggCol, 1].clone()

                idx = torch.where(result <= target_result)[0]
                bs_l[idx] = legal_tensors[idx, aggCol, 1].clone()
            elif binary_side =='left':
                # 1 - result is the `result` for binary_side=='right'
                idx = torch.where(1 - result  > target_result)[0]
                bs_r[idx] = legal_tensors[idx, aggCol, 1].clone()

                idx = torch.where(1 - result  <= target_result)[0]
                bs_l[idx] = legal_tensors[idx, aggCol, 1].clone()
            else:
                assert False

    # compute final result
    with torch.no_grad():
        result = (bs_l + bs_r)/2
        result = DW.GetDeNormalizedValue(aggCol, result)
        Delta = torch.Tensor([DW.delta[aggCol]])

        if DW.delta[aggCol] != 0:
            result = torch.floor(result / Delta) * Delta
        result = result.to('cpu')

        n_ = numQuery
        oracle_list = oracle_cards.copy()
    
    # Below is the same as other agg functions
    err_list = BatchErrorMetrix(result, oracle_list)
    with HiddenPrints():
        if ifGroup is False:
            groupTime = 0

        total_query_time = total_time + groupTime

        avg_per_query_time = 1000. * (total_query_time / n_)
        avg_group_time = 1000. * groupTime / n_
        avg_f_batch_time = 1000. * f_batch_time / n_

        avg_vegas_time = avg_per_query_time - avg_f_batch_time - avg_group_time

    print("********** total_n=[{}] batchn=[{}]  N=[{}]  nitr=[{}]  alpha=[{}]  beta=[{}] ******".format(n_, n, N,
                                                                                                        num_iterations,
                                                                                                        alpha, beta))
    print('@ Average per query          [{}] ms'.format(avg_per_query_time))

    print(' --  Average per query group [{}] ms'.format(avg_group_time))
    print(' --  Average per query NF    [{}] ms'.format(avg_f_batch_time))
    print(' --  Average per query vegas [{}] ms'.format(avg_vegas_time))
    p50 = np.percentile(err_list, 50)
    p95 = np.percentile(err_list, 95)
    p99 = np.percentile(err_list, 99)
    pmax = np.max(err_list)
    mean = np.average(err_list)
    print('Mean [{:.3f}]  Median [{:.3f}]  95th [{:.3f}]  99th [{:.3f}]  max [{:.3f}]'.format(mean, p50, p95,
                                                                                              p99, pmax))

    return p50, p95, p99, pmax, mean, result




def transform_MODE_queries(DW, queries):
    """ Transform mode queries to COUNT queries """
    new_queries = []
    agg_col_values = [] # literals over the agg column
    mapping = []

    # aggCol info
    aggColName = DW.columns[aggCol]
    aggColMin = DW.Mins[aggCol]
    aggColMax = DW.Maxs[aggCol]
    aggColDomain = int(aggColMax - aggColMin + 1)
    aggColDomainValues = DW.data.iloc[:, aggCol].unique()
    aggColDomainValues.sort()

    num_count_queries = 0 # accumulated number of  count queries

    for query in queries:
        # check if aggColName is in cols of the query
        cols, ops, vals = query
        aggColIdx = -1
        for i,col in enumerate(cols):
            if col == aggColName:
                aggColIdx = i
                break

        # aggCol exists in existing cols
        if aggColIdx != -1:
            op = ops[aggColIdx]
            val = vals[aggColIdx]
            # compute all the values in the domain that satisfies op val condition
            available_values = OPS[op](aggColDomainValues, val)
            available_values = aggColDomainValues[available_values]
            for aggValue in available_values:
                _ops = ops.copy()
                _vals = vals.copy()
                _vals[aggColIdx] = aggValue
                _ops[aggColIdx] = '='
                new_query = (cols, _ops, _vals)
                new_queries.append(new_query)
                agg_col_values.append(aggValue)

            mapping.append((num_count_queries, num_count_queries + available_values.shape[0]))
            num_count_queries += available_values.shape[0]

        else:   # aggCol does not exist in existing cols
            newCols = cols.append(np.take(DW.columns,[aggCol]))
            newOps = np.append(ops, '=')
            for aggValue in range(int(aggColMin), int(aggColMax) + 1):
                newVals = np.append(vals, aggValue)
                new_query = (newCols, newOps, newVals)
                new_queries.append(new_query)
                agg_col_values.append(aggValue)
            mapping.append((num_count_queries, num_count_queries + aggColDomain))
            num_count_queries += aggColDomain

    return new_queries, agg_col_values, mapping


def estimate_MODE_results(result, MODE_mapping, COUNT_aggCol_values):
    # estimate MODE results based on the estimation of the COUNT queries
    results = []
    # compute the argmax of the result for the range in MODE_mapping
    for interval in MODE_mapping:
        start, end = interval
        # get the index of the last value in result[start:end]
        max_countid = torch.argmax(result[start:end])
        mode = COUNT_aggCol_values[start + max_countid]
        results.append(mode)
    return results
