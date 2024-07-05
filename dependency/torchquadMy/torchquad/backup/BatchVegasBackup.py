"""
    2022.5.18
    Batch version of vegas

    TODO List:
        - v1：最简单版本 for循环单个正常积分执行 ✅
            -- insight: 这里要实现的 BatchVegas 和 vegas.py里的作用其实是一样的… 只是要多存几个map stratification
        - v2：将求积分的过程提取出来。让多个VEGAS sample的点只调用一次flow ✅
            -- insight: 效果惊人。代码尚有待进一步优化，但表现已经很行了…
            -- 一些待加/优化的：  alpha，beta的设定 ✅
        - v3: 用大的map初始化新的map ✅
            -- in vegas_map.py:
                --- 可能需要一个get_Y() nope UPD: ✅
                --- 需要该的就是 dx_edges, x_edges 其他变量的是确定的 ✅
                --- transfer_map的过程需要优化，
        - v4：将VEGAS的更新过程也变为vectorized的同时做
"""
from autoray import numpy as anp
from autoray import infer_backend, astype
from loguru import logger


from .base_integrator import BaseIntegrator
from .utils import _setup_integration_domain
from .vegas import VEGAS
import torch
from .rng import RNG
from .vegas_map import VEGASMap
from .vegas_stratification import VEGASStratification

class BatchVEGAS(BaseIntegrator):

    # Number of different vegas integrators
    _n = None

    # Number of sampling points used for each integrator
    _N = None

    # Integration domain list  (backend tensor)
    _integration_domains = None



    def __init__(self):
        super().__init__()

    def initList(self):
        """ initialize map, strat list """
        if self._n is not None:
            self.map_list = []
            self.strat_list = []
            self.results_list = []
            self.sigma2_list = []
        else:
            raise ValueError("self._n not set! Please run")


    def setValues(self,
                  fn,
                  dim,
                  N,
                  n,
                  integration_domains,
                  iterations= 20,
                  backend="torch",
                  dtype="float32",
                  eps_rel=0,
                  alpha=0.5,
                  beta=0.75,
                  target_map=None,
                  target_domain_starts = None,
                  target_domain_sizes = None,
                  reuse_sample_points=False,
                  rng=None,
                  seed=None,
                  ):
        """ Initialize required attributes"""
        self._fn = fn
        self._dim = dim
        self._N = N
        self._n = n
        self._integration_domains = integration_domains

        self._iterations = iterations

        # 不确定这样是不是就ok了
        self._backend = backend
        self._dtype = integration_domains[0].dtype
        if rng is None:
            rng = RNG(backend=self._backend, seed=seed)
        elif seed is not None:
            raise ValueError("seed and rng cannot both be passed")
        self.rng = rng

        self._N_increment = N // (self._iterations + 5)
        self._starting_N_val = N // (self._iterations + 5)
        self._starting_N = anp.repeat(
            self._starting_N_val, self._n
        )
        self.use_grid_improve = True

        self._eps_rel = eps_rel
        self._alpha = alpha
        self._beta = beta

        """ variables related to transfer map"""
        self.target_map = target_map
        self.reuse_sample_points = reuse_sample_points
        self.target_domain_starts = target_domain_starts
        self.target_domain_sizes = target_domain_sizes

        self.initList()



    def integrate(self):
        """ v1 最简单版本 for循环单个正常积分执行 """
        vegas = VEGAS()
        res = torch.zeros(self._n, device=self._integration_domains.device)
        for i in range(self._n):
            res[i] = vegas.integrate(self._fn,
                                     dim = self._dim,
                                     N = self._N,
                                     integration_domain=self._integration_domains[i,:],
                                     use_warmup=False,
                                     use_grid_improve=True)
        return res

    def integrate1(self):
        self._check_inputs(dim=self._dim,
                           N=self._N,
                           integration_domain=self._integration_domains[0,:])
        integration_domain = _setup_integration_domain(self._dim,
                                                       self._integration_domains[0,:],
                                                       self._backend)
        # print("Integration domain is ")
        # print(integration_domain)


        """ TODO: 这块改成 记录每个的transform参数"""
        self._domain_starts_list  = []
        self._domain_sizes_list   = []
        self._domain_volume_list = []

        """ 确定 domain相关信息 """
        for integration_domain in self._integration_domains:
            # Transform the integrand into the [0,1]^dim domain
            domain_starts = integration_domain[:, 0]
            domain_sizes = integration_domain[:, 1] - domain_starts
            domain_volume = anp.prod(domain_sizes)

            self._domain_starts_list.append(domain_starts)
            self._domain_sizes_list.append(domain_sizes)
            self._domain_volume_list.append(domain_volume)

            # print("i= ",i)
            # print(domain_starts)
            # print(domain_sizes)
            # print(domain_volume)

        """ TODO: 后面在eval前 把这个x转换一下 """
        # def transformed_integrand(x):
        #     return fn(x * domain_sizes + domain_starts) * domain_volume

        # self._fn = transformed_integrand  # TODO: 这里得更改，现在这样不太行


        """ TODO： 这块改成 建立多个"""

        """ 建立 map & strat"""
        for i in range(self._n):
            # Initialize the adaptive VEGAS map,
            # Note that a larger number of intervals may lead to problems if only few evals are allowed
            # Paper section II B
            N_intervals = max(2, self._N_increment // 10)  # for small N intervals set 2


            self.map_list.append(VEGASMap(N_intervals, self._dim, self._backend, self._dtype,alpha=self._alpha))

            # Initialize VEGAS' stratification
            # Paper section III
            self.strat_list.append(VEGASStratification(
                self._N_increment,
                dim=self._dim,
                rng=self.rng,
                backend=self._backend,
                dtype=self._dtype,
                beta=self._beta
            ))

            logger.debug("Starting VEGAS")

            self.results_list.append([])  # contains integration results per iteration
            self.sigma2_list.append([])  # contains variance per iteration

        self.it = 0  # iteration


        """ TODO: 现在这里全是拿同一个map初始化 """

        if self.reuse_sample_points:
            self.map_list = self.transfer_map(self.map_list, self.target_map)

        # Main loop
        while True:

            self.it = self.it + 1
            for i in range(self._n):
                self.results_list[i].append(0)
                self.sigma2_list[i].append(0)

            # Compute current iteration for every integrator
            self._run_iteration()

            # logger.info(
            #     f"Iteration {self.it}, Acc={acc:.4e}, Result={self.results[-1]:.4e},neval={self._nr_of_fevals}"
            # )
            if self.it >= self._iterations:
                break


        # logger.info(
        #     f"Computed integral after {self._nr_of_fevals} evals was {self._get_result():.8e}."
        # )
        return self._get_result_list()
        # return self._get_result()

    def _generate_sampling_points(self):
        """ (1) generate sampling points for each integrator """
        self.n_eval_list = []
        self.y_list = []
        self.x_list = []
        self.siz_list = []
        for i in range(self._n):
            neval = self.strat_list[i].get_NH(self._starting_N[i])  # Evals per strat cube

            # Stratified sampling points y and transformed sample points x
            y = self.strat_list[i].get_Y(neval)
            x = self.map_list[i].get_X(y)  # transform, EQ 8+9
            x = (x * self._domain_sizes_list[i] + self._domain_starts_list[i])
            # fn(x * domain_sizes + domain_starts) * domain_volume

            self.n_eval_list.append(neval)
            self.y_list.append(y)
            self.x_list.append(x)
            self.siz_list.append(x.shape[0])


    def _evaluate_sampling_points(self):
        """ (2) evaluate all the sampling points using NF
            the results are split into self.f_eval_list
        """
        self.f_eval_list = []
        inp = torch.cat(self.x_list)

        f_eval = self._eval(inp).squeeze()
        self.f_eval_list = torch.split(f_eval, self.siz_list)


        # pass

    def _update_each_integrator(self):
        """ (3) update each integrator """
        for i in range(self._n):
            f_eval = self.f_eval_list[i] * self._domain_volume_list[i]
            jac = self.map_list[i].get_Jac(self.y_list[i])  # compute jacobian
            jf_vec = f_eval * jac  # precompute product once
            jf_vec2 = jf_vec**2
            if self._backend == "torch":
                jf_vec2 = jf_vec2.detach()

            if self.use_grid_improve:  # if adaptive map is used, acc weight
                self.map_list[i].accumulate_weight(self.y_list[i], jf_vec2)  # EQ 25

            jf, jf2 = self.strat_list[i].accumulate_weight(self.n_eval_list[i], jf_vec)  # update strat

            neval_inverse = 1.0 / astype(self.n_eval_list[i], self.y_list[i].dtype)
            ih = jf * (neval_inverse * self.strat_list[i].V_cubes)  # Compute integral per cube

            # Collect results
            sig2 = jf2 * neval_inverse * (self.strat_list[i].V_cubes**2) - pow(ih, 2)
            if self._backend == "torch":
                sig2 = sig2.detach()
            # Sometimes rounding errors produce negative values very close to 0
            sig2 = anp.abs(sig2)
            self.results_list[i][-1] = ih.sum()  # store results
            self.sigma2_list[i][-1] = (sig2 * neval_inverse).sum()

            if self.use_grid_improve:  # if on, update adaptive map
                logger.debug("Running grid improvement")
                self.map_list[i].update_map()
            self.strat_list[i].update_DH()  # update stratification

            # Estimate an accuracy for the logging
            acc = anp.sqrt(self.sigma2_list[i][-1])
            if self.results_list[i][-1] != 0.0:
                acc = acc / anp.abs(self.results_list[i][-1])

        # return acc

        # pass

    def _enhance_iterations(self):
        """ Every fifth iteration reset the sample integral results
        and adjust the number of evaluations per iteration

        Returns:
            Bool: True iff VEGAS should abort
        """
        # Abort only every fifth iteration
        if self.it % 5 > 0 or self.it >= self._iterations:
            return False

        # Abort conditions depending on achieved errors
        self.res_abs = anp.abs(self._get_result_list())
        err_list = self._get_error()
        chi2_list = self._get_chisq()

        # Adjust number of evals if Chi square indicates instability
        # EQ 32
        for i in range(self._n):
            if chi2_list[i] / 5.0 < 1.0:
                # Use more points in the next iterations to reduce the
                # relative error
                if self.res_abs[i] == 0.0:
                    self._starting_N[i] += self._N_increment
                else:
                    acc = err_list[i] / self.res_abs[i]
                    self._starting_N[i] = min(
                        self._starting_N[i] + self._N_increment,
                        int(self._starting_N[i] * anp.sqrt(acc / (self._eps_rel + 1e-8))),
                    )
            elif chi2_list[i] / 5.0 > 1.0:
                # Use more points in the next iterations because of instability
                self._starting_N[i] += self._N_increment

                """ TODO: 把这个加回去 """
                # Abort if the next 5 iterations would use too many function
                # evaluations
                # if self._nr_of_fevals + self._starting_N * 5 > self.N:
                #     return True

                self.results_list[i] = []  # reset sample results
                self.sigma2_list[i] = []  # reset sample results
        return False



    def _run_iteration(self):
        """Runs one iteration of VEGAS including stratification and updates the VEGAS map if use_grid_improve is set.

        TODO: divide into 3 parts:
            - generate sampling points for each integrator
            - evaluate all the sampling points using NF
            - update each integrator

        Returns:
            backend-specific float: Estimated accuracy.
        """
        self._generate_sampling_points()
            # self.n_eval_list.append(neval)
            # self.y_list.append(y)
            # self.x_list.append(x)
            # self.siz_list.append(x.shape[0])
        self._evaluate_sampling_points()
        self._update_each_integrator()
        self._enhance_iterations()
        return

    #
    # Helper funcs
    def _get_result_list(self):
        """Computes mean of results to estimate integral, EQ 30.

        Returns:
            backend-specific float: Estimated integral.
        """
        res_list = torch.Tensor(self._n)
        for i in range(self._n):

            if any(sig2 == 0.0 for sig2 in self.sigma2_list[i]):
                # If at least one variance is 0, return the mean result
                res = sum(self.results_list[i]) / len(self.results_list[i])
            else:
                res_num = sum(res / sig2 for res, sig2 in zip(self.results_list[i], self.sigma2_list[i]))
                res_den = sum(1.0 / sig2 for sig2 in self.sigma2_list[i])
                res = res_num / res_den

            if self._backend == "numpy" and res.dtype != self.results_list[i][0].dtype:
                # NumPy automatically casts float32 to float64 in the above
                # calculations
                res = astype(res, self.results_list[i][0].dtype)
            res_list[i] = res
        return res_list



    def _get_error(self):
        """Estimates error from variance , EQ 31.

        Returns:
            backend-specific float: Estimated error.

        """
        # Skip variances which are zero and return a backend-specific float
        res_list = torch.Tensor(self._n)
        for i in range(self._n):
            res = sum(1.0 / sig2 for sig2 in self.sigma2_list[i] if sig2 != 0.0)
            res_list[i] = self.sigma2_list[i][0] if res == 0 else 1.0 / anp.sqrt(res)
        return res_list

    def _get_chisq_i(self, i):
        """Computes chi square from estimated integral and variance, EQ 32.

        Returns:
            backend-specific float: Chi squared.
        """
        """ assert """
        I_final = self.res_abs[i]
        return sum(
            (
                (res - I_final) ** 2 / sig2
                for res, sig2 in zip(self.results_list[i], self.sigma2_list[i])
                if res != I_final
            ),
            start=self.results_list[i][0] * 0.0,
        )

    def _get_chisq(self):
        """Computes chi square from estimated integral and variance, EQ 32.

        Returns:
            backend-specific float: Chi squared.
        """
        res_list = torch.Tensor(self._n)
        for i in range(self._n):
            res_list[i] = self._get_chisq_i(i)
        return res_list


    """ new helper function 2: calculate dx_edges, x_edges for 【y_lims】(a sub-range of [0,1] according to this map  """

    def get_dx_x_edges(self, target_map, n_edges, y_lims):
        """
        Args:
            target_map (VEGASMap)   : target map that this range(y_lims) should be transformed from
            n_interval (backend int): number of edges
            y_lims (backend tensor):  range limits (lower bound & upper bound) for each dimension   size: 2 * self._dim

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
        # grids = torch.linspace(z[:,0],z[:,1],100)
        # torch.linspace()

    """ new helper function 3: re-initialize maps in map_list using full_map  """
    def transfer_map(self, map_list, full_map):
        """

        Args:
            map_list (a list of VEGASMap) : maps with uniform initialization
            full_map (VEGASMap) : a map converged for full integration domain, use this to adjust map_list
                                    TODO: in the future, this should be changed into a list or something

        Returns:
            transfered_map_list (a list of VEGASMap)

        step 1: get y_range for each integrator using  self._integration_domains
                【note】this should be conducted in vectorization

                self._integration_domains -> x
                x -> (x - target_domain_starts) / target_domain_sizes
                感觉还是得有y才行啊 现在这样还是只是x
                TODO: x -> y,  i.e.,  full_map.getY(x)     need getY ✅

        【a simple yet not very effective method】
        dx_edges, x_edges = self.get_dx_x_edges(full_map)
        map_list[i].set_map_edges(dx_edges, x_edges)

        【standard approach】
        step 2: linspace each y
                y -> torch.linspace(y,100)

        step3 : getX for each linspaced y
                y -> x'    i.e., full_map.getY(x)

        step4 : normalize the result x into [0,1]
        """
        # print("self._integration_domains is ")
        # print(self._integration_domains.shape)
        # print(self._integration_domains)
        # ys = self.target_map.get_Y(self._integration_domains)

        # TODO: vectorization
        for i in range(len(map_list)):
            # 20,10,2    n, dim, min|max
            # if i==0:
            #     continue
            """ TODO： 为啥hui yo"""
            this_x = self._integration_domains[i,:].T
            this_x = (this_x - self.target_domain_starts) / self.target_domain_sizes

            """ 临时去掉 """
            ys = full_map.get_Y(this_x).T
            # print("see ys")
            # print("ys.shape ", ys.shape)
            # print(ys)

            _dx_edges, _x_edges = self.get_dx_x_edges(full_map, map_list[i].N_intervals + 1, ys)

            # print("see dx edges")
            # print(_dx_edges)
            map_list[i].set_map_edges(_dx_edges.T, _x_edges.T)
        return map_list
        # step 1: get y_range for each integrator using  self._integration_domains
        #         【note】this should be conducted in vectorization
        #
        #         self._integration_domains -> x
        #         x -> (x - target_domain_starts) / target_domain_sizes
        #         感觉还是得有y才行啊 现在这样还是只是x
        #         TODO: x -> y,  i.e.,  full_map.getY(x)     need getY