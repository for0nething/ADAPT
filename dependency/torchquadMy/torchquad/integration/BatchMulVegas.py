"""
    2022.6.1
    Batch version of vegas with  vegas_mul_map  & hcube
    这个Batch指的是 map 完全用 vegas_mul_map 替换，vectorize 地同时做n个map的操作 版本的batch
    hcube 部分，暂时还是维护一个list

    TODO List:
        - v1：vegas_mul_map + vegas_mul_stratification
            -- insight: 用vegas_mul_map 的 vectorize操作替代vegas_map
            -- 用 vegas_mul_stratification 代替 vegas_stratification
            -- 这部分还是用 BatchVegasBackup 里 比较老的版本的改的。因为新版的更新还分了slow fast， 没有必要
            -- 先验证 vegas_mul_map 和 vegas_mul_stratification 的正确性
            --
            --

"""

from autoray import numpy as anp
from autoray import infer_backend, astype
from loguru import logger
import time

from .base_integrator import BaseIntegrator
from .utils import _setup_integration_domain
from .vegas import VEGAS
import torch
from .rng import RNG
from .vegas_mul_map import VEGASMultiMap
from .vegas_mul_stratification import VEGASMultiStratification


class BatchMulVEGAS(BaseIntegrator):
    # Number of different vegas integrators
    _n = None

    # Number of sampling points used for each integrator
    _N = None

    # Integration domain list  (backend tensor)
    _integration_domains = None

    def __init__(self):
        super().__init__()


    def setValues(self,
                  fn,
                  dim,
                  N,
                  n,
                  integration_domains,
                  iterations=20,
                  backend="torch",
                  dtype="float32",
                  eps_rel=0,
                  alpha=0.5,
                  beta=0.75,
                  target_map=None,
                  target_domain_starts=None,
                  target_domain_sizes=None,
                  reuse_sample_points=False,
                  rng=None,
                  seed=None,
                  useMask=False,
                  mask_integration_domains=None,
                  integration_weights=None
                  ):
        """ Initialize required attributes"""
        self._fn = fn
        self._dim = dim
        self._N = N
        self._n = n
        # (n, dim, 2)
        self._integration_domains = integration_domains

        self._iterations = iterations
        self.it = 0

        # 不确定这样是不是就ok了
        self._backend = backend
        self._dtype = integration_domains[0].dtype
        if rng is None:
            rng = RNG(backend=self._backend, seed=seed)
        elif seed is not None:
            raise ValueError("seed and rng cannot both be passed")
        self.rng = rng


        """ map & strat 相关的参数"""

        # 原本版本
        # self._N_increment = N // (self._iterations + 5)
        # self._starting_N_val = N // (self._iterations + 5)
        # self._starting_N = self._starting_N_val
        # self.N_intervals = max(2, self._N_increment // 10)  # for small N intervals set 2


        # 修改版本
        self._N_increment = N // (self._iterations )
        self._starting_N_val = N // (self._iterations )
        self._starting_N = self._starting_N_val
        self.N_intervals = max(2, self._N_increment // 10)  # for small N intervals set 2
        # if self.N_intervals > 100:
        #     self.N_intervals = 100


        self.use_grid_improve = True
        self._eps_rel = eps_rel
        self._alpha = alpha
        self._beta = beta

        """ 一些用于得到结果的变量 """
        self.results = torch.zeros((self._iterations, self._n))
        self.sigma2  = torch.zeros((self._iterations, self._n))

        """ variables related to transfer map"""
        self.target_map = target_map
        self.reuse_sample_points = reuse_sample_points
        # (dim)
        self.target_domain_starts = target_domain_starts
        # (dim)
        self.target_domain_sizes = target_domain_sizes

        self.useMask = useMask
        self.mask_integration_domains = mask_integration_domains
        if self.useMask:
            assert self.mask_integration_domains is not None
            maskN = self.mask_integration_domains.shape[0]
            self.maskN = maskN
            self.results = torch.zeros((self._iterations, maskN))
            self.sigma2 = torch.zeros((self._iterations, maskN))
            # 确定有多少个大的integration
            self.clusterN = self._integration_domains.shape[0]
            # 确定每个大的integration （对于这里时integration_domains里的) 的weight
            self.integration_weights = integration_weights
            # 没有weight的话 weights直接设置为maskN
            if self.integration_weights is None:
                self.integration_weights = torch.tensor([maskN], dtype=int)
            else:
                assert self.integration_weights.shape[0] == self.clusterN
            # 初始化 maskN  ->  clusterN
            self.integration_mapping  = torch.arange(self.clusterN)
            # self.integration_mapping  = torch.repeat_interleave(self.integration_mapping,
            #                                                     torch.tensor(self.integration_weights,
            #                                                                  device=self.integration_mapping.device,
            #                                                                  dtype=int))
            self.integration_mapping = torch.repeat_interleave(self.integration_mapping,
                                                               self.integration_weights.clone().detach().int())
            self.masks= None

    def integrate(self):
        self._check_inputs(dim=self._dim,
                           N=self._N,
                           integration_domain=self._integration_domains[0,:])


        """ TODO: 这块改成 记录每个的transform参数"""
        # self._domain_starts = torch.empty((self._dim, self._n))
        self._domain_starts = torch.empty((self._n, self._dim))
        self._domain_sizes = torch.empty_like(self._domain_starts)
        self._domain_volume = torch.empty((self._n))

        """ 确定 domain相关信息 """
        # TODO： 改输入 之后变成batch的做
        for i, integration_domain in enumerate(self._integration_domains):
            # Transform the integrand into the [0,1]^dim domain
            domain_starts = integration_domain[:, 0]
            domain_sizes = integration_domain[:, 1] - domain_starts
            domain_volume = anp.prod(domain_sizes)
            # (n, dim)
            self._domain_starts[i,:] = domain_starts
            self._domain_sizes[i,:]  = domain_sizes
            # (n)
            self._domain_volume[i]   = domain_volume

        # (dim,n)
        self._domain_starts = self._domain_starts.permute(1,0)
        self._domain_sizes = self._domain_sizes.permute(1,0)
        # (dim, n, 1)
        self._domain_starts = self._domain_starts.unsqueeze(2)
        self._domain_sizes = self._domain_sizes.unsqueeze(2)
        # (n, 1)
        self._domain_volume = self._domain_volume.unsqueeze(1)



        """ 建立 map & strat"""
        self.map = VEGASMultiMap(
            n=self._n,
            alpha=self._alpha,
            N_intervals=self.N_intervals,
            dim=self._dim,
            backend=self._backend,
            dtype=self._dtype)

        self.strat = VEGASMultiStratification(
            n=self._n,
            N_increment=self._N_increment,
            dim=self._dim,
            rng=self.rng,
            backend=self._backend,
            dtype=self._dtype,
            beta=self._beta)

        """ transfer integration map (or group map) by target map """
        if self.reuse_sample_points:
            # self.transfer_map(self.target_map)
            self.transfer_map_vec(self.target_map)

        # Main loop
        while True:
            print("#"*40)
            print("    Current iteration is {}".format(self.it))
            print("#"*40)
            # Compute current iteration for every integrator
            if self.useMask:
                self._run_iteration_mask()
            else:
                self._run_iteration()
            self.it = self.it + 1
            if self.it >= self._iterations:
                break

        return self._get_result()

    def _run_iteration(self):
        """Runs one iteration of VEGAS including stratification and updates the VEGAS map if use_grid_improve is set.

             Divide into 3 parts:
            - ✅ generate sampling points for each integrator
            - ✅ evaluate all the sampling points using NF
            - ✅ update each integrator
            - ✅ Enhance
        Returns:
            backend-specific float: Estimated accuracy.
        """
        self._generate_sampling_points()
        self._evaluate_sampling_points()
        self._update_each_integrator()

        # 先不enhance了
        # self._enhance_iterations()
        return

    def _run_iteration_mask(self):
        """Runs one iteration of VEGAS including stratification and updates the VEGAS map if use_grid_improve is set.

            Divide into 3 parts:
            - ✅ generate sampling points for each integrator
            - ✅ evaluate all the sampling points using NF
            - ✅ update each integrator
            - ✅ Enhance
        Returns:
            backend-specific float: Estimated accuracy.
        """

        self._generate_sampling_points()
        self._evaluate_sampling_points()
        self._update_each_integrator_mask()

        # 先不enhance了
        # self._enhance_iterations()
        return

    def _generate_sampling_points(self):
        """ ✅(1) generate sampling points for each integrator """
        # (n, N_cubes)
        self.nevals = self.strat.get_NH(self._starting_N)


        # (n, bla, dim)
        self.y = self.strat.get_Y(self.nevals)
        # (dim, n, bla)
        self.y = self.y.permute(2, 0, 1)
        # (dim, n, bla)  [0,1]的
        self.x = self.map.get_X(self.y)

        # (dim, n, bla)  变回真正范围的
        self.x = (self.x * self._domain_sizes + self._domain_starts)


    def _evaluate_sampling_points(self):
        """ ✅(2) evaluate all the sampling points using NF
            the results are split into self.f_eval_list
        """
        # (n, bla ,dim)
        inp = self.x.permute(1,2,0)

        # (total_n, dim)
        inp = inp.view(-1, inp.shape[-1])

        # (total_n, 1)
        self.f_eval = self._eval(inp)

        # (n, bla)
        self.f_eval = self.f_eval.view(self._n, -1)
        print("Classic f_eval ", self.f_eval.max(), self.f_eval.median())

    def _update_each_integrator(self):
        # self.f_eval shape:  (n, bla)
        self.f_eval = self.f_eval * self._domain_volume
        # (n, bla)
        jac = self.map.get_Jac(self.y)
        jf_vec = self.f_eval * jac
        jf_vec2 = jf_vec ** 2
        jf_vec2 = jf_vec2.detach()

        if self.use_grid_improve:
            self.map.accumulate_weight(self.y, jf_vec2)

        # (n, N_cubes)
        jf, jf2 = self.strat.accumulate_weight(self.nevals, jf_vec)  # update strat
        # (n, N_cubes)
        neval_inverse = 1.0 / astype(self.nevals, self.y.dtype)

        ih = jf * (neval_inverse * self.strat.V_cubes)  # Compute integral per cube

        # Collect results
        sig2 = jf2 * neval_inverse * (self.strat.V_cubes ** 2) - pow(ih, 2)
        sig2 = sig2.detach()

        # Sometimes rounding errors produce negative values very close to 0
        sig2 = anp.abs(sig2)
        self.results[self.it, :] = ih.sum(axis=1)  # store results
        self.sigma2[self.it, :] = (sig2 * neval_inverse).sum(axis=1)

        if self.use_grid_improve:  # if on, update adaptive map
            logger.debug("Running grid improvement")
            self.map.update_map()

        self.strat.update_DH()  # update stratification

        # Estimate an accuracy for the logging
        # acc = anp.sqrt(self.sigma2[self.it, :])
        # if torch.count_nonzero(self.results[self.it,:]) == self._n:
        #     acc = acc/ anp.abs(self.results[self.it,:])


    def _get_error(self):
        """Estimates error from variance , EQ 31.

        Returns:
            backend-specific float: Estimated error.

        """
        # Skip variances which are zero and return a backend-specific float
        # TODO： 如果有sig2 是0的会出问题
        res = torch.sum(1./self.sigma2, axis=1)
        res = 1./anp.sqrt(res)
        return res


    def _get_chisq(self):
        """Computes chi square from estimated integral and variance, EQ 32.

        Returns:
            backend-specific float: Chi squared.
        """

        use_results = self.results.clone()
        use_results[0,:] = 0 # 第一轮的全部设置乘0

        use_results = use_results.permute(1,0)
        use_sigma2 = self.sigma2.clone()
        use_sigma2 = use_sigma2.permute(1,0)

        res = torch.sum(
            (use_results - self.res_abs) ** 2 / use_sigma2, axis=1
        )
        return res


    def _enhance_iterations(self):
        """ Every fifth iteration reset the sample integral results
                and adjust the number of evaluations per iteration

                Returns:
                    Bool: True iff VEGAS should abort
                """
        # Abort only every fifth iteration
        if self.it % 5 > 0 or self.it >= self._iterations or self.it <=0:
            return False

        # Abort conditions depending on achieved errors
        # self.res_abs = anp.abs(self._get_result())
        # err_list = self._get_error()
        # chi2_list = self._get_chisq()

        # Adjust number of evals if Chi square indicates instability
        # EQ 32
        """ TODO: 这里也有问题， 现在只能每个分别处理… 然而starting_N还没有统一orz """
        self._starting_N += self._N_increment
        # for i in range(self._n):
        #     if chi2_list[i] / 5.0 < 1.0:
        #         # Use more points in the next iterations to reduce the
        #         # relative error
        #         if self.res_abs[i] == 0.0:
        #             self._starting_N[i] += self._N_increment
        #         else:
        #             acc = err_list[i] / self.res_abs[i]
        #             self._starting_N[i] = min(
        #                 self._starting_N[i] + self._N_increment,
        #                 int(self._starting_N[i] * anp.sqrt(acc / (self._eps_rel + 1e-8))),
        #             )
        #     elif chi2_list[i] / 5.0 > 1.0:
        #         # Use more points in the next iterations because of instability
        #         self._starting_N[i] += self._N_increment
        #
        #         """ TODO: 把这个加回去 """
        #         # Abort if the next 5 iterations would use too many function
        #         # evaluations
        #         # if self._nr_of_fevals + self._starting_N * 5 > self.N:
        #         #     return True
        #
        #         self.results_list[i] = []  # reset sample results
        #         self.sigma2_list[i] = []  # reset sample results
        return False

    def _get_result(self):
        # TODO: 某一个有 0 就不能这么搞了… 姑且先这样？
        results = self.results.permute(1,0) # (nitr, n) -> (n, nitr)
        sigma2 = self.sigma2.permute(1,0)

        res_num = torch.sum(results/sigma2, dim=1)
        res_den = torch.sum(1./sigma2, dim=1)
        res = res_num / res_den
        return res

    """ helper functions (Deserted)"""
    def get_dx_x_edges(self, target_map, n_edges, y_lims):
        """
        Args:
            target_map (VEGASMap)   : target map that this range(y_lims) should be transformed from
            n_interval (backend int): number of edges
            y_lims (backend tensor):  range limits (lower bound & upper bound) for each dimension   size: (self._dim, 2)

        Returns:
            dx_edges (backend tensor) : dx of edges for each dimension   size: (n_edges - 1) * self._dim
            x_edges  (backend tensor) : edges for each dimension         size: n_edges * self._dim
        """
        ret_x_edges  = torch.empty((n_edges, self._dim))
        # ret_x_edges = torch.empty((self._dim, n_edges))


        for i in range(self._dim):
            if y_lims[i,1] > 0.999999:    # 偶买噶… 这个太大会有问题
                y_lims[i,1] = 0.999999
            ret_x_edges[:,i] = torch.linspace(y_lims[i, 0], y_lims[i, 1], n_edges)

        # todo: 这里可能有问题
        ret_x_edges = target_map.get_X(ret_x_edges)

        ret_dx_edges = ret_x_edges[1:, :] - ret_x_edges[:-1, :]

        """ 每一维度得normalize到 [0,1] 还是不对 得 (? -min)/size """

        _tmp_max, _ = ret_x_edges.max(dim = 0)
        _tmp_min, _ = ret_x_edges.min(dim = 0)
        siz = _tmp_max - _tmp_min

        # siz = siz / (self.target_domain_sizes)
        ret_x_edges = (ret_x_edges - _tmp_min) / siz
        ret_dx_edges = ret_dx_edges / siz

        return ret_dx_edges, ret_x_edges


    def get_dx_x_edges_vec(self, target_map, n_edges, y_lims):
        """
        Args:
            target_map (VEGASMap)   : target map that this range(y_lims) should be transformed from
            n_interval (backend int): number of edges
            y_lims (backend tensor):  range limits (lower bound & upper bound) for each dimension   size: (n, 2, self._dim)

        Returns:
            dx_edges (backend tensor) : dx of edges for each dimension   size: (n_edges - 1) * self._dim
            x_edges  (backend tensor) : edges for each dimension         size: n_edges * self._dim
        """
        # (n, n_edges, dim)
        ret_x_edges  = torch.empty((self._n, n_edges, self._dim))
        # ret_x_edges = torch.empty((self._dim, n_edges))

        # (n, 2, dim)
        y_lims[:, 1, :] = torch.clamp(y_lims[:, 1, :], max=0.999999)

        # 可能还是跑不了一个linspace
        # TODO: 优化
        n = self._integration_domains.shape[0]
        for i in range(n):
            for j in range(self._dim):
                ret_x_edges[i, :, j] = torch.linspace(y_lims[i, 0, j], y_lims[i, 1, j], n_edges)

        ret_x_edges = ret_x_edges.view(-1, self._dim)
        # todo: 这里可能有问题   vegas_map ( n * n_edges, DIM)
        ret_x_edges = target_map.get_X(ret_x_edges)

        ret_x_edges = ret_x_edges.view(self._n, n_edges, self._dim)
        ret_dx_edges = ret_x_edges[:,1:,:] -  ret_x_edges[:, :-1, :]
        # ret_dx_edges = ret_x_edges[1:, :] - ret_x_edges[:-1, :]
        assert ret_dx_edges.shape == (self._n, n_edges-1, self._dim)

        """ 每一维度得normalize到 [0,1] 还是不对 得 (? -min)/size """

        _tmp_max, _ = ret_x_edges.max(dim = 1)
        _tmp_min, _ = ret_x_edges.min(dim = 1)
        # print("see tmp_max shape", _tmp_max.shape)
        _tmp_max = _tmp_max.view(ret_x_edges.shape[0], 1, ret_x_edges.shape[2])
        _tmp_min = _tmp_min.view(ret_x_edges.shape[0], 1, ret_x_edges.shape[2])

        siz = _tmp_max - _tmp_min

        # print("shapes ", ret_x_edges.shape, _tmp_max.shape, _tmp_min.shape, siz.shape)
        # siz = siz / (self.target_domain_sizes)
        ret_x_edges = (ret_x_edges - _tmp_min) / siz
        ret_dx_edges = ret_dx_edges / siz

        return ret_dx_edges, ret_x_edges

    """ new helper function 3: re-initialize maps in map_list using full_map  """
    def transfer_map(self, full_map):
        # TODO: vectorization
        for i in range(self._n):
            # 20,10,2    n, dim, min|max
            # if i==0:
            #     continue
            """ TODO： 为啥hui yo"""

            this_x = self._integration_domains[i, :].T   # (2, dim)
            this_x = (this_x - self.target_domain_starts) / self.target_domain_sizes # (2,dim)

            """ 临时去掉 """
            ys = full_map.get_Y(this_x).T
            # print("see ys")
            # print("ys.shape ", ys.shape)
            # print(ys)

            _dx_edges, _x_edges = self.get_dx_x_edges(full_map, self.N_intervals + 1, ys)

            self.map.set_map_edges_i(i, dx_edges=_dx_edges, x_edges=_x_edges)

        return


    # TODO: 这个写对了就可以把上面的非vec的去掉了
    def transfer_map_vec(self, full_map):
        """ vectorize version of transfer_map"""
        # 该函数内的n 应该是group对应的n
        n = self._integration_domains.shape[0]
        # 最初 integration_domains.shape ==  (n, dim, 2)
        # permute完 变为 (n, 2, dim)
        x_lims = self._integration_domains.permute(0, 2, 1)
        # target_xxx 均为 (dim)
        x_lims = (x_lims - self.target_domain_starts) / self.target_domain_sizes # (n, 2, dim)

        # TODO: 这里可能得优化下 现在就先怎么方便怎么来了
        # (2n, dim)
        x_lims = x_lims.reshape(-1, x_lims.shape[-1])
        # 这个是 vegas_map里写的

        # (2n, dim)
        y_lims = full_map.get_Y(x_lims)
        _dx_edges, _x_edges = self.get_dx_x_edges_vec(full_map, self.N_intervals + 1, y_lims.view(n, 2, -1))


        self.map.set_map_edges(dx_edges=_dx_edges, x_edges=_x_edges)

        print(x_lims)


    def getMask(self, maskDomain, N_strat=None):
        """ 由 mapDomain的strat 到 maskDomain 的strat 计算strat的mask应该是多少

            maskDomain:  要计算mask的domain范围 (dim, 2)
            N_strat:     strat每一维度的上升个数
        """
        # (dim, 2)
        x_lims = (maskDomain - self.target_domain_starts.view(-1,1)) / self.target_domain_sizes.view(-1,1)

        # TODO： maskDomain得先转换为y space。 这个mask是每一轮都得变的

        # (dim, 2)
        maskDomain = self.target_map.get_Y(x_lims.T).T

        # stratification steps per dim, computed from EQ 41
        # stepNum = self.strat.N_strat
        if N_strat is not None:
            stepNum = N_strat
        else:
            stepNum = self.strat.N_strat

        # number of dims
        d = self._dim
        # step 1. compute mask per dim
        mask = torch.zeros((d, stepNum))

        # iterate each dimension, compute mask for each dimension
        for dim in range(d):
            # 1.1 compute scales       stepNum +1 个刻度 将区间分成stepNum份
            # scales = torch.linspace(mapDomain[dim, 0], mapDomain[dim, 1], stepNum + 1,
            scales = torch.linspace(0, 1, stepNum + 1,
                                    dtype=self._dtype)
            # lim_l, lim_r 分别和maskDomain[dim,0], maskDomain[dim,1] 取max 和 min
            # mask[dim, :] = (lim_r - lim_l) / stepSize
            scales.detach()
            lim_l = scales[:-1].clone()
            lim_r = scales[1:].clone()

            lim_l[lim_l < maskDomain[dim, 0]] = maskDomain[dim, 0]
            lim_r[lim_r > maskDomain[dim, 1]] = maskDomain[dim, 1]
            stepSize = scales[1] - scales[0]

            mask[dim, :] = (lim_r - lim_l) / stepSize
        mask[mask < 0] = 0
        # step 2. compute Cartesian product of 'mask per dim' in step 1, in this way, get final mask
        # 【Notice】 这个笛卡尔积好像还得反过来… 似乎。 因为现在的mul_strat按维度从低到高，是从低位往高位来的
        # 反过来一下
        mask = torch.flip(mask, dims=[0])
        z = torch.split(mask, 1, dim=0)
        mask = [t.squeeze(dim=0) for t in z]
        mask = torch.meshgrid(mask, indexing='ij')
        mask = torch.stack(mask, dim=-1)
        # mask = torch.cartesian_prod((mask[i,:] for i in range(mask.shape[0])))

        mask = torch.prod(mask, dim=-1)
        return mask



    def getMaskVec(self, maskDomains, N_strat=None):
        """ vectorize 版的 getMask

            maskDomains:  要计算mask的domains范围 (n, dim, 2)  需要先转换为 (dim, 2n)
            N_strat:     strat每一维度的上升个数
        """
        """ 相当于全用getMaskVecToMulti了 """
        if self.clusterN >= 1:
            return self.getMaskVecToMulti(maskDomains, N_strat)
        n = maskDomains.shape[0]

        # (n, dim, 2)  -> (dim, n, 2)
        maskDomains = maskDomains.permute(1, 0, 2)
        # (dim, n, 2)  -> (dim, 2n)
        maskDomains = maskDomains.reshape(maskDomains.shape[0], -1)

        # (dim, 2n)
        x_lims = (maskDomains - self.target_domain_starts.view(-1, 1)) / self.target_domain_sizes.view(-1, 1)
        # (dim, 2n) 这步似乎是vectorize的 TODO: 有待检查
        st_time  = time.time()

        x_limsT = (x_lims.T).contiguous()
        maskDomains = self.target_map.get_Y(x_limsT).T
        print("get_Y took ", time.time() - st_time)
        # (dim, n, 2)
        maskDomains = maskDomains.view(maskDomains.shape[0], n, -1)

        # stratification steps per dim, computed from EQ 41
        # stepNum = self.strat.N_strat
        if N_strat is not None:
            stepNum = N_strat
        else:
            stepNum = self.strat.N_strat

        # number of dims
        d = self._dim
        # step 1. compute mask per dim

        # (dim, n, stepNum)
        # masks = torch.zeros((d, n, stepNum))

        stepSize = torch.empty(d)
        lim_l = torch.empty(d, n, stepNum)
        lim_r = torch.empty(d, n, stepNum)

        st_time = time.time()
        # iterate each dimension, compute mask for each dimension
        for dim in range(d):
            # 1.1 compute scales       stepNum +1 个刻度 将区间分成stepNum份
            # scales = torch.linspace(mapDomain[dim, 0], mapDomain[dim, 1], stepNum + 1,
            scales = torch.linspace(0, 1, stepNum + 1,
                                    dtype=self._dtype)
            stepSize[dim] = scales[1] - scales[0]

            scales.detach()
            lim_l[dim,:] = scales[:-1].clone().unsqueeze(0).repeat(n, 1)
            lim_r[dim,:] = scales[1:].clone().unsqueeze(0).repeat(n, 1)

            # """ TODO: 这个for循环可以通过repeat maksDomain去掉"""
            # # 1。 复制maskDomains   (dim, n, stepNum)
            # # 2。 确定小的index 改值
            # # 3。 确定大的index 改值
            # for i in range(n):
            #     lim_l[dim, i, :] = torch.clamp(lim_l[dim, i, :], min=maskDomains[dim, i, 0])
            #     lim_r[dim, i, :] = torch.clamp(lim_r[dim, i, :], max=maskDomains[dim, i, 1])

        # (dim, n, stepNum)
        # min
        broadDomain = maskDomains[:, :, 0].unsqueeze(-1).repeat(1, 1, stepNum)
        less_index = lim_l < broadDomain
        lim_l[less_index] = broadDomain[less_index]
        # max
        broadDomain = maskDomains[:, :, 1].unsqueeze(-1).repeat(1, 1, stepNum)
        larger_index = lim_r > broadDomain
        lim_r[larger_index] = broadDomain[larger_index]


        print("Lim for took ",time.time()- st_time)
        stepSize = stepSize.view(-1, 1, 1)
        masks = (lim_r - lim_l) / stepSize

        # (dim, n, stepNum)
        masks[masks < 0] = 0
        # (n, dim, stepNum)
        masks = masks.permute(1, 0, 2)
        # step 2. compute Cartesian product of 'mask per dim' in step 1, in this way, get final mask

        # (n, nhcubes, dim)
        tmp = torch.empty(masks.shape[0], masks.shape[2]**masks.shape[1], d )
        # 这部分现在只能for循环来做 TODO：找一下有没有vec的方法


        # 遍历n 当n大了估计就慢了 TODO: 优化这个… 可能有点难

        masks = torch.flip(masks, dims=[1])
        st_time = time.time()
        for i in range(n):
            # 这个dims=[0] 得确认一下对不对
            # mask = torch.flip(masks[i, :], dims=[0])

            mask = masks[i, :]
            z = torch.split(mask, 1, dim=0)
            mask = [t.squeeze(dim=0) for t in z]
            mask = torch.meshgrid(mask, indexing='ij')
            # masks[i, :] = torch.stack(mask, dim=-1)
            tmp[i, :] = torch.stack(mask, dim=-1).view(-1, d)


        print("Meshgrid took ", time.time() - st_time)
        # (n, nhcube, dim)
        # masks = torch.stack(masks, dim=-1)
        # (n, nhcube, dim) -> (n, nhcube, 1)
        masks = torch.prod(tmp, dim=-1)
        # masks = torch.prod(masks, dim=-1)
        return masks

    def getMaskVecToMulti(self, maskDomains, N_strat=None):
        """
            TODO 有待实现
            【多对多】
            vectorize 版的 getMask

            maskDomains:  要计算mask的domains范围 (n, dim, 2)  需要先转换为 (dim, 2n)
            N_strat:     strat每一维度的上升个数
        """
        n = maskDomains.shape[0]

        # (n, dim, 2)  -> (dim, n, 2)
        maskDomains = maskDomains.permute(1, 0, 2)
        # (dim, n, 2)  -> (dim, 2n)
        maskDomains = maskDomains.reshape(maskDomains.shape[0], -1)

        # TODO: 这得改成按照对应的target_domain算
        # (dim, 2n)

        # (1) 计算 对应的大domain的domain_starts, domain_size
        # domain_starts.shape   (dim, clusterN)
        # 因为是区间 所以得乘2
        # (dim,n,1)  ->  (dim, n)
        self._domain_starts = self._domain_starts.squeeze(-1)
        self._domain_sizes = self._domain_sizes.squeeze(-1)

        """ 这块得把domain_starts domain_size 修改为group对应的取值"""
        _domain_starts = torch.repeat_interleave(self._domain_starts, 2 * self.integration_weights, dim=1)
        _domain_sizes  = torch.repeat_interleave(self._domain_sizes, 2 * self.integration_weights, dim=1)

        self._domain_starts = self._domain_starts.unsqueeze(-1)
        self._domain_sizes = self._domain_sizes.unsqueeze(-1)
        # (2) 算x_lims
        # x_lims = (maskDomains - self.target_domain_starts.view(-1, 1)) / self.target_domain_sizes.view(-1, 1)
        x_lims = (maskDomains - _domain_starts) / _domain_sizes
        # (dim, 2n) 这步似乎是vectorize的
        st_time  = time.time()

        # maskDomains = self.target_map.get_Y(x_lims.T).T
        # x_limsT = (x_lims.T).contiguous()
        # TODO: 这里的target_map得改成 MulMap 并且需要实现其中的get_Y函数
        # …… 这里错了
        x_lims = x_lims.view(self._dim, n, -1)
        maskDomains = self.map.get_y(x_lims, self.integration_mapping)
        # maskDomains = self.target_map.get_Y(x_limsT).T
        print("get_Y took ", time.time() - st_time)
        # (dim, n, 2)
        maskDomains = maskDomains.view(maskDomains.shape[0], n, -1)

        # stratification steps per dim, computed from EQ 41
        # stepNum = self.strat.N_strat
        if N_strat is not None:
            stepNum = N_strat
        else:
            stepNum = self.strat.N_strat

        # number of dims
        d = self._dim
        # step 1. compute mask per dim

        # (dim, n, stepNum)
        # masks = torch.zeros((d, n, stepNum))

        stepSize = torch.empty(d)
        lim_l = torch.empty(d, n, stepNum)
        lim_r = torch.empty(d, n, stepNum)

        st_time = time.time()
        # iterate each dimension, compute mask for each dimension
        for dim in range(d):
            # 1.1 compute scales       stepNum +1 个刻度 将区间分成stepNum份
            # scales = torch.linspace(mapDomain[dim, 0], mapDomain[dim, 1], stepNum + 1,
            scales = torch.linspace(0, 1, stepNum + 1,
                                    dtype=self._dtype)
            stepSize[dim] = scales[1] - scales[0]

            scales.detach()
            lim_l[dim,:] = scales[:-1].clone().unsqueeze(0).repeat(n, 1)
            lim_r[dim,:] = scales[1:].clone().unsqueeze(0).repeat(n, 1)

            # """ TODO: 这个for循环可以通过repeat maksDomain去掉"""
            # # 1。 复制maskDomains   (dim, n, stepNum)
            # # 2。 确定小的index 改值
            # # 3。 确定大的index 改值
            # for i in range(n):
            #     lim_l[dim, i, :] = torch.clamp(lim_l[dim, i, :], min=maskDomains[dim, i, 0])
            #     lim_r[dim, i, :] = torch.clamp(lim_r[dim, i, :], max=maskDomains[dim, i, 1])

        # (dim, n, stepNum)
        # min
        # TODO： 这里的broadDomain得按照对应关系分别算出来
        broadDomain = maskDomains[:, :, 0].unsqueeze(-1).repeat(1, 1, stepNum)
        less_index = lim_l < broadDomain
        lim_l[less_index] = broadDomain[less_index]
        # max
        broadDomain = maskDomains[:, :, 1].unsqueeze(-1).repeat(1, 1, stepNum)
        larger_index = lim_r > broadDomain
        lim_r[larger_index] = broadDomain[larger_index]


        print("Lim for took ",time.time()- st_time)
        stepSize = stepSize.view(-1, 1, 1)
        masks = (lim_r - lim_l) / stepSize

        # (dim, n, stepNum)
        masks[masks < 0] = 0
        # (n, dim, stepNum)
        masks = masks.permute(1, 0, 2)
        # step 2. compute Cartesian product of 'mask per dim' in step 1, in this way, get final mask

        # (n, nhcubes, dim)
        tmp = torch.empty(masks.shape[0], masks.shape[2]**masks.shape[1], d )

        # 这部分现在只能for循环来做 TODO：找一下有没有vec的方法
        # 遍历n 当n大了估计就慢了 TODO: 优化这个… 可能有点难
        masks = torch.flip(masks, dims=[1])
        st_time = time.time()
        for i in range(n):
            # 这个dims=[0] 得确认一下对不对
            # mask = torch.flip(masks[i, :], dims=[0])

            mask = masks[i, :]
            z = torch.split(mask, 1, dim=0)
            mask = [t.squeeze(dim=0) for t in z]
            mask = torch.meshgrid(mask, indexing='ij')
            # masks[i, :] = torch.stack(mask, dim=-1)
            tmp[i, :] = torch.stack(mask, dim=-1).view(-1, d)


        print("Meshgrid took ", time.time() - st_time)
        # (n, nhcube, dim)
        # masks = torch.stack(masks, dim=-1)
        # (n, nhcube, dim) -> (n, nhcube, 1)
        masks = torch.prod(tmp, dim=-1)
        # masks = torch.prod(masks, dim=-1)
        return masks

    def _update_each_integrator_mask(self):
        """ 2022.08.25
            使用mask进行计算
        """
        assert self.mask_integration_domains is not None


        self.f_eval = self.f_eval * self._domain_volume
        # (n, bla)
        jac = self.map.get_Jac(self.y)
        jf_vec = self.f_eval * jac
        jf_vec2 = jf_vec ** 2
        jf_vec2 = jf_vec2.detach()

        if self.use_grid_improve:
            self.map.accumulate_weight(self.y, jf_vec2)

        # (n, N_cubes)
        jf, jf2 = self.strat.accumulate_weight(self.nevals, jf_vec)  # update strat
        # (n, N_cubes)
        neval_inverse = 1.0 / astype(self.nevals, self.y.dtype)

        ih = jf * (neval_inverse * self.strat.V_cubes)  # Compute integral per cube

        # Collect results
        sig2 = jf2 * neval_inverse * (self.strat.V_cubes ** 2) - pow(ih, 2)
        sig2 = sig2.detach()

        # Sometimes rounding errors produce negative values very close to 0
        sig2 = anp.abs(sig2)

        # TODO： 只需要修改这里
        # (1) 逐一计算mask
        # (2) 用mask算results
        # mask = self.getMask(maskDomain = )
        # masks = torch.empty_like(ih)
        # 每一轮重新算mask
        # n = self.mask_integration_domains.shape[0]


        st = time.time()
        # TODO: 把这个改成vectorize的
        if self._alpha ==0:
            if self.masks is None:
                self.masks = self.getMaskVec(self.mask_integration_domains)
            masks = self.masks
        else:
            masks = self.getMaskVec(self.mask_integration_domains)
        # for i in range(n):
        #     masks[i, :] = self.getMask(self.mask_integration_domains[i,:]).view(-1)
        en = time.time()
        print("Compute mask took ", en - st)
        # print("masks max", masks.max())
        print("masks shape is ", masks.shape)
        ih = torch.repeat_interleave(ih, self.integration_weights, dim=0)
        self.results[self.it, :] = (ih * masks).sum(axis=1)  # store results

        # self.results[self.it, :] = ih.sum(axis=1)  # store results


        # self.sigma2[self.it, :] = (sig2 * neval_inverse).sum(axis=1)
        tmp_sigma2 = (sig2 * neval_inverse).sum(axis=1)
        self.sigma2[self.it, :] = torch.repeat_interleave(tmp_sigma2, self.integration_weights, dim=0)


        if self.use_grid_improve:  # if on, update adaptive map
            logger.debug("Running grid improvement")
            self.map.update_map()

        self.strat.update_DH()  # update stratification








