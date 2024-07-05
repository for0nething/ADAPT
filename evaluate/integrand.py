from parameterSetting import *
import torch
import time



def get_f_batch(flow, DW):
    """ get f_batch function for different aggregate functions """
    global f_batch_time

    # COUNT
    def f_batch_count(inp):
        global f_batch_time
        with torch.no_grad():
            inp = inp.cuda()

            print("【Example input】", inp[0, :])
            print("inp shape ", inp.shape)
            print("inp dtype ", inp.dtype)
            st = time.time()
            prob_list = flow.log_prob(inp)
            prob_list = torch.exp(prob_list)

            # avoid nan
            small_padding = 1e-20
            prob_list[torch.isnan(prob_list)] = small_padding

            print("【nan num】 ", torch.isnan(prob_list).sum())

            #         print("shape is ", prob_list.shape)
            print("【max_prob】 ", prob_list.max())
            print("【median_prob】 ", prob_list.median())
            en = time.time()
            f_batch_time += en - st

            return prob_list


    def f_batch_sum(inp, dim=aggCol):
        global f_batch_time
        assert dim is not None
        with torch.no_grad():
            inp = inp.cuda()

            print("【Example input】", inp[0, :])
            print("inp shape ", inp.shape)
            print("inp dtype ", inp.dtype)
            st = time.time()
            prob_list = flow.log_prob(inp)


            agg_inp = DW.GetDeNormalizedValue(dim, inp[:, dim])
            Delta = torch.Tensor([DW.delta[dim]])

            if DW.delta[dim] != 0:
                agg_inp = torch.floor(agg_inp / Delta) * Delta
            #         agg_inp = inp[:, dim]

            prob_list = torch.exp(prob_list)


            small_padding = 1e-20
            prob_list[torch.isnan(prob_list)] = small_padding

            print("【nan num】 ", torch.isnan(prob_list).sum())

            prob_list = prob_list * agg_inp
            print("【max_prob】 ", prob_list.max())
            print("【median_prob】 ", prob_list.median())
            en = time.time()
            f_batch_time += en - st

            return prob_list



    def f_batch_square_sum(inp, dim=aggCol):
        global f_batch_time
        assert dim is not None
        with torch.no_grad():
            inp = inp.cuda()

            print("【Example input】", inp[0, :])
            print("inp shape ", inp.shape)
            print("inp dtype ", inp.dtype)
            st = time.time()
            prob_list = flow.log_prob(inp)


            agg_inp = DW.GetDeNormalizedValue(dim, inp[:, dim])
            Delta = torch.Tensor([DW.delta[dim]])

            if DW.delta[dim] != 0:
                agg_inp = torch.floor(agg_inp / Delta) * Delta
            #         agg_inp = inp[:, dim]

            prob_list = torch.exp(prob_list)


            small_padding = 1e-20
            prob_list[torch.isnan(prob_list)] = small_padding

            print("【nan num】 ", torch.isnan(prob_list).sum())

            prob_list = prob_list * agg_inp * agg_inp
            print("【max_prob】 ", prob_list.max())
            print("【median_prob】 ", prob_list.median())
            en = time.time()
            f_batch_time += en - st

            return prob_list



    # select aggregate func
    if agg_type == 'count':
        f_batch = f_batch_count
    elif agg_type =='sum':
        f_batch = f_batch_sum
    elif agg_type =='average':
        f_batch = f_batch_sum
    elif agg_type =='variance':
        f_batch = f_batch_square_sum
    elif agg_type == 'percentile':
        f_batch = f_batch_count
    elif agg_type == 'mode':
        f_batch = f_batch_count
    elif agg_type == 'range':
        f_batch = f_batch_count

    return f_batch, f_batch_count, f_batch_sum, f_batch_square_sum