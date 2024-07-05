from parameterSetting import *
import time
import torch
from torchquad import  enable_cuda, VEGAS
import pickle


def reuse(DW, f_batch):
    global REUSE_FROM_FILE
    if REUSE_FROM_FILE == True:
        try:
            f = open(REUSE_FILE_PATH + '{}.pickle'.format(dataset_name), 'rb')
            target_map = pickle.load(f)
        except:
            REUSE_FROM_FILE = False


    z = DW.getLegalRangeQuery([[],[],[]])
    # z = torch.Tensor(z)
    # print(z.shape)
    full_integration_domain = torch.tensor(z, dtype=torch.float)
    # full_integration_domain = torch.DoubleTensor(z).cuda()

    domain_starts = full_integration_domain[:, 0]
    domain_sizes =  full_integration_domain[:, 1] - domain_starts
    domain_volume = torch.prod(domain_sizes)


    if REUSE_FROM_FILE == False:
        vegas = VEGAS()
        bigN = 500000 * 40

        st = time.time()
        result = vegas.integrate(f_batch,dim=features,
                                 N=bigN,
                                 integration_domain=full_integration_domain,
                                 use_warmup=True,
                                 use_grid_improve=True,
                                 max_iterations=40
                                 )

        en= time.time()
        print("Took ", en-st)
        print(result)
        result = result * DW.n

        print('result is ',result)


    if REUSE_FROM_FILE == False:
        target_map = vegas.map
        import pickle
        f=open(REUSE_FILE_PATH + "{}.pickle".format(dataset_name),'wb')
        pickle.dump(target_map, f)
        f.close()

    return target_map, domain_starts, domain_sizes