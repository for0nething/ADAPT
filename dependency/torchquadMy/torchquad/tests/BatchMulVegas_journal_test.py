import sys

sys.path.append("../")
import time
from autoray import numpy as anp
from autoray import to_backend_dtype, astype
import timeit
import cProfile
import pstats
import torch
from unittest.mock import patch

from integration.vegas import VEGAS
from integration.rng import RNG
from integration.BatchMulVegas import BatchMulVEGAS

import integration.utils as utils

""" 只需要改动这里就能在cpu和gpu之间切换"""
# import os
# #cpu版
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# DEVICE = "cpu"

#gpu版
DEVICE= "cuda"


from helper_functions import (
    compute_integration_test_errors,
    setup_test_for_backend,
    get_test_functions

)


# def f_batch(inp):
#     inp[:,2] = inp[:,2] * 4 + 2
#     return torch.sum(inp, axis=1) * 10

def f_batch(inp):
    inp[:,2] = inp[:,2] * 4 + 2
    inp[:, -2:] = inp[:, -2:] **2
    # inp[:, -3] = inp[:, -3]
    # inp = inp**2
    # inp = inp/
    return torch.sum(inp, axis=1)


# 阶段一：  单个query  用 单个大的query  的sample points
def _run_one_to_one_BatchMulVegas():
    z = BatchMulVEGAS()

    # parameters setting
    dim = 10
    n = 200
    N = 200000

    # (dim, 2)
    full_integration_domain = torch.Tensor(dim * [[0,1]])

    # (1) fullmap  并且让fullmap 收敛
    vegas = VEGAS()
    bigN = 1000000 * 40
    st = time.time()
    result = vegas.integrate(f_batch, dim=dim,
                             N=bigN,
                             integration_domain=full_integration_domain,
                             use_warmup=True,
                             use_grid_improve=True,
                             max_iterations=40
                             )
    en = time.time()
    print("Took ", en - st)
    print(result)
    result = result
    print('【full domain】 result is ', result)


    domain_starts = full_integration_domain[:, 0]
    domain_sizes = full_integration_domain[:, 1] - domain_starts

    # (2) 确定小的domain
    samll_domain_head = torch.Tensor(5 * [[0, 0.8]])
    small_domain_tail = torch.Tensor(5 * [[0.3, 1]])
    # (dim, 2)
    small_domain = torch.concat((samll_domain_head, small_domain_tail))



    # (3) 计算小domain的实际值
    sm_vegas = VEGAS()
    bigN = 1000000 * 10

    st = time.time()
    result = sm_vegas.integrate(f_batch, dim=dim,
                             N=bigN,
                             integration_domain=small_domain,
                             use_warmup=True,
                             use_grid_improve=True,
                             max_iterations=10
                             )
    en = time.time()
    print("Took ", en - st)
    print(result)
    small_result = result
    print('【small domain】 real result is ', small_result)

    # (4) 计算用mask算的小domian的值 这个可能比较难
    # (4.1) 先直接试试看  单独调用get_mask 好不好用 ✅ 但不知道mask对不对

    # mask = z.getMask(small_domain, N_strat=2)
    # print("#"*20)
    # print("Mask shape")
    # print(mask.shape)

    # (4.2) 整个过程用mask来算
    z.setValues(f_batch,
                dim=dim,
                N=N,
                n=1,
                # integration_domains=small_domain.unsqueeze(0),
                integration_domains=full_integration_domain.unsqueeze(0),
                rng=None,
                seed=1234,
                reuse_sample_points=True,
                mask_integration_domains=small_domain.unsqueeze(0),
                target_map=vegas.map,
                target_domain_sizes=domain_sizes,
                target_domain_starts=domain_starts,
                useMask=True
                )

    maskResult = z.integrate()

    z.setValues(f_batch,
                dim=dim,
                N=N,
                n=1,
                integration_domains=small_domain.unsqueeze(0),
                rng=None,
                seed=1234,
                reuse_sample_points=False,
                target_map=vegas.map,
                target_domain_sizes=domain_sizes,
                target_domain_starts=domain_starts,
                useMask=False
                )
    nomaskResult = z.integrate()
    print('【small domain】 real result is ', small_result)
    print('【No Mask】 result is ', nomaskResult)
    print('【Mask】 result is ', maskResult)
    # (5) 评估误差


# 阶段二：  多个query  用 单个大的query  的sample points
def _run_mul_to_one_BatchMulVegas():
    z = BatchMulVEGAS()

    # parameters setting
    dim = 10
    n = 200
    N = 200000

    # (dim, 2)
    full_integration_domain = torch.Tensor(dim * [[0,2]])

    # (1) fullmap  并且让fullmap 收敛
    # (2) 确定小的domain
    # (3) 计算小domain的实际值
    # (4) 计算用mask算的小domian的值
    # (5) 评估误差

    # (1) ✅fullmap  并且让fullmap 收敛
    vegas = VEGAS()
    bigN = 1000000 * 40

    st = time.time()
    result = vegas.integrate(f_batch, dim=dim,
                             N=bigN,
                             integration_domain=full_integration_domain,
                             use_warmup=True,
                             use_grid_improve=True,
                             max_iterations=40
                             )
    en = time.time()
    print("Took ", en - st)
    print(result)
    result = result
    print('【full domain】 result is ', result)


    domain_starts = full_integration_domain[:, 0]
    domain_sizes = full_integration_domain[:, 1] - domain_starts

    # (2) ✅确定小的domain
    samll_domain_head = torch.Tensor(5 * [[0, 1.8]])
    small_domain_tail = torch.Tensor(5 * [[0, 1.2]])
    # (dim, 2)
    small_domain = torch.concat((samll_domain_head, small_domain_tail))
    small_domain = small_domain.unsqueeze(0)
    # 区别在于 复制多份
    small_domain = small_domain.repeat(n, 1, 1)

    ano_head = torch.Tensor(4*[[0, 1.75]])
    ano_tail = torch.Tensor(6*[[0.1, 0.9]])
    ano = torch.concat((ano_head, ano_tail))
    small_domain[-1,:] = ano

    ano_head = torch.Tensor(4 * [[1, 1.4]])
    ano_tail = torch.Tensor(6 * [[0.1, 1.2]])
    ano = torch.concat((ano_head, ano_tail))
    small_domain[-2, :] = ano

    ano_head = torch.Tensor(5 * [[0, 2]])
    ano_tail = torch.Tensor(5 * [[0, 2]])
    ano = torch.concat((ano_head, ano_tail))
    small_domain[-3, :] = ano

    z.setValues(f_batch,
                dim=dim,
                N=N,
                n=1,
                # integration_domains=small_domain.unsqueeze(0),
                integration_domains=full_integration_domain.unsqueeze(0),
                rng=None,
                seed=1234,
                reuse_sample_points=True,
                iterations=5,
                # mask_integration_domains=small_domain.unsqueeze(0),
                mask_integration_domains=small_domain,
                target_map=vegas.map,
                target_domain_sizes=domain_sizes,
                target_domain_starts=domain_starts,
                useMask=True
                )



    # (3) 计算小domain的实际值
    sm_vegas = BatchMulVEGAS()
    bigN = 10000

    st = time.time()
    sm_vegas.setValues(f_batch,
                        dim=dim,
                        n=n,
                        N=bigN,
                        integration_domains=small_domain,
                        rng=None,
                        seed=1234,
                        reuse_sample_points=True,
                        target_map=vegas.map,
                        target_domain_sizes=domain_sizes,
                        target_domain_starts=domain_starts,
                     )
    result = sm_vegas.integrate()
    en = time.time()
    print("Took ", en - st)
    print(result)
    small_result = result
    print('【small domain】 real result is ', small_result)

    # (4) 计算用mask算的小domian的值 这个可能比较难
    # (4.1) 先直接试试看  单独调用get_mask 好不好用 ✅ 但不知道mask对不对

    # mask = z.getMask(small_domain, N_strat=2)
    # print("#"*20)
    # print("Mask shape")
    # print(mask.shape)

    # (4.2) 整个过程用mask来算
    maskResult = z.integrate()


    z.setValues(f_batch,
                dim=dim,
                N=N,
                n=n,
                integration_domains=small_domain,
                rng=None,
                seed=1234,
                reuse_sample_points=False,
                target_map=vegas.map,
                target_domain_sizes=domain_sizes,
                target_domain_starts=domain_starts,
                useMask=False
                )
    nomaskResult = z.integrate()
    print('【small domain】 real result is ', small_result)
    print('【No Mask】 result is ', nomaskResult)
    print('【Mask】 result is ', maskResult)
    # (5) 评估误差



# 阶段三：  多个query  用 多个大的query  的sample points
def _run_mul_to_mul_BatchMulVegas():
    z = BatchMulVEGAS()

    # parameters setting
    dim = 10
    n = 200
    N = 200000

    # (dim, 2)
    full_integration_domain = torch.Tensor(dim * [[0,2]])

    # (1) fullmap  并且让fullmap 收敛   得加mul_map的计算
    #     传的integration_domain 得是所有的大的domain
    #      ✅ <phase 1> 让integration domain 就是所有要求的domain, i.e., 所有的mask都是1, get_Y里的参数是arange(N)
    #      <phase 2> 真的把query聚类
    # (2) ✅确定小的domain            【不用改】
    # (3) ✅计算小domain的实际值       【不用改】
    # (4) 计算用mask算的小domian的值      得改成新的
    # (5) 评估误差                    【不用改】

    # (1) ✅fullmap  并且让fullmap 收敛
    vegas = VEGAS()
    bigN = 1000000 * 40

    st = time.time()
    result = vegas.integrate(f_batch, dim=dim,
                             N=bigN,
                             integration_domain=full_integration_domain,
                             use_warmup=True,
                             use_grid_improve=True,
                             max_iterations=40
                             )
    en = time.time()
    print("Took ", en - st)
    print(result)
    result = result
    print('【full domain】 result is ', result)
    domain_starts = full_integration_domain[:, 0]
    domain_sizes = full_integration_domain[:, 1] - domain_starts





    # (2) ✅确定小的domain
    samll_domain_head = torch.Tensor(5 * [[0, 1.8]])
    small_domain_tail = torch.Tensor(5 * [[0, 1.2]])
    # (dim, 2)
    small_domain = torch.concat((samll_domain_head, small_domain_tail))
    small_domain = small_domain.unsqueeze(0)
    # 区别在于 复制多份
    small_domain = small_domain.repeat(n, 1, 1)

    # 修改一部分small_domain的取值
    ano_head = torch.Tensor(4*[[0, 1.75]])
    ano_tail = torch.Tensor(6*[[0.1, 0.9]])
    ano = torch.concat((ano_head, ano_tail))
    small_domain[-1,:] = ano

    ano_head = torch.Tensor(4 * [[1, 1.4]])
    ano_tail = torch.Tensor(6 * [[0.1, 1.2]])
    ano = torch.concat((ano_head, ano_tail))
    small_domain[-2, :] = ano

    ano_head = torch.Tensor(5 * [[0, 2]])
    ano_tail = torch.Tensor(5 * [[0, 2]])
    ano = torch.concat((ano_head, ano_tail))
    small_domain[-3, :] = ano


    # 设定integration_weights
    _integration_weights = torch.ones(n, dtype=int)
    z.setValues(f_batch,
                dim=dim,
                N=N,
                n=n,
                integration_domains=small_domain,
                # integration_domains=full_integration_domain.unsqueeze(0),
                rng=None,
                seed=1234,
                reuse_sample_points=True,
                iterations=5,
                # mask_integration_domains=small_domain.unsqueeze(0),
                mask_integration_domains=small_domain,
                target_map=vegas.map,
                target_domain_sizes=domain_sizes,
                target_domain_starts=domain_starts,
                useMask=True,
                integration_weights=_integration_weights
                )

    

    # (3) 计算小domain的实际值
    sm_vegas = BatchMulVEGAS()
    bigN = 10000

    st = time.time()
    sm_vegas.setValues(f_batch,
                        dim=dim,
                        n=n,
                        N=bigN,
                        integration_domains=small_domain,
                        rng=None,
                        seed=1234,
                        reuse_sample_points=True,
                        target_map=vegas.map,
                        target_domain_sizes=domain_sizes,
                        target_domain_starts=domain_starts,
                     )
    result = sm_vegas.integrate()
    en = time.time()
    print("Took ", en - st)
    print(result)
    small_result = result
    print('【small domain】 real result is ', small_result)

    # (4) 计算用mask算的小domian的值 这个可能比较难
    # (4.1) 先直接试试看  单独调用get_mask 好不好用 ✅ 但不知道mask对不对
    # (4.2) 整个过程用mask来算
    maskResult = z.integrate()
    #
    #
    # z.setValues(f_batch,
    #             dim=dim,
    #             N=N,
    #             n=n,
    #             integration_domains=small_domain,
    #             rng=None,
    #             seed=1234,
    #             reuse_sample_points=False,
    #             target_map=vegas.map,
    #             target_domain_sizes=domain_sizes,
    #             target_domain_starts=domain_starts,
    #             useMask=False
    #             )
    # nomaskResult = z.integrate()


    
    z.setValues(f_batch,
                dim=dim,
                N=N,
                n=1,
                # integration_domains=small_domain,
                integration_domains=full_integration_domain.unsqueeze(0),
                rng=None,
                seed=1234,
                reuse_sample_points=True,
                iterations=5,
                # mask_integration_domains=small_domain.unsqueeze(0),
                mask_integration_domains=small_domain,
                target_map=vegas.map,
                target_domain_sizes=domain_sizes,
                target_domain_starts=domain_starts,
                useMask=True,
                integration_weights=torch.tensor([small_domain.shape[0]], dtype=int)
                )
    many_to_one_result = z.integrate()

    """ 【new】 多对多的测试 """
    z.setValues(f_batch,
                dim=dim,
                N=N,
                n=2,
                # integration_domains=small_domain,
                integration_domains=full_integration_domain.unsqueeze(0).repeat(2,1,1),
                rng=None,
                seed=1234,
                reuse_sample_points=True,
                iterations=5,
                # mask_integration_domains=small_domain.unsqueeze(0),
                mask_integration_domains=small_domain,
                target_map=vegas.map,
                target_domain_sizes=domain_sizes,
                target_domain_starts=domain_starts,
                useMask=True,
                integration_weights=torch.tensor([int(small_domain.shape[0]/2)], dtype=int).repeat(2)
                )
    many_to_many_result = z.integrate()

    print('【small domain】 real result is ', small_result)
    # print('【No Mask】 result is ', nomaskResult)
    print('【Mask】 result is ', maskResult)
    print('【Many to one 】 result is ', many_to_one_result)
    print("【many to many】 result is ", many_to_many_result)

    # (5) 评估误差


# 实际跑的函数
def _run_BatchVegas_tests(backend, dtype_name):
    utils.add_time = 0
    st = time.time()
    """Test if VEGAS+ works with example functions and is accurate as expected"""
    # _run_vegas_accuracy_checks(backend, dtype_name)

    # 阶段一：  单个query  用 单个大的query  的sample points
    # _run_one_to_one_BatchMulVegas()

    # 阶段二：  多个query  用 单个大的query  的sample points
    # _run_mul_to_one_BatchMulVegas()

    # 阶段三：  多个query  用 多个大的query  的sample points
    _run_mul_to_mul_BatchMulVegas()

    print("")
    print("Total add_time is ", utils.add_time)
    en = time.time()
    print("Total time is ", en-st)
    # _run_example_integrations(backend, dtype_name)


test_integrate_torch = setup_test_for_backend(_run_BatchVegas_tests, "torch", "float64")


if __name__ == "__main__":
    # used to run this test individually
    # test_integrate_numpy()

    profile_torch = False

    if profile_torch:
        profiler = cProfile.Profile()
        profiler.enable()
        start = timeit.default_timer()
        test_integrate_torch()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("tottime")
        stats.print_stats()
        stop = timeit.default_timer()
        print("Test ran for ", stop - start, " seconds.")
    else:
        test_integrate_torch()

