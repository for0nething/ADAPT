import numpy as np


import torch
from dataUtils import *
from parameterSetting import *
from scipy import stats as st
from consts import filterNum


class DataWrapper():
    def __init__(self, data, dataset_name):

        self.data = data
        self.n = data.shape[0]
        self.cardinality = data.shape[0]
        self.dim = data.shape[1]
        self.columns = data.columns
        self.dataset_name = dataset_name

        self.Mins = data.min(axis=0)
        self.Maxs = data.max(axis=0)

        self.minFilter, self.maxFilter = filterNum[self.dataset_name]
        self.sensible_to_do_range = sensible[self.dataset_name]
        self.colMap = {col: i for i, col in enumerate(self.columns)}
        self.delta = deltas[self.dataset_name]

    def getColID(self, col):
        return self.colMap[col]

    def getCateColumns(self, cols):
        cols = [self.getColID(col) for col in cols]
        return self.sensible_to_do_range[cols]

    def GetNormalizedValue(self, col_id, val):
        if self.dataset_name == 'power':
            return val
        U = Norm_us[self.dataset_name]
        S = Norm_ss[self.dataset_name]
        ret = (val - U[col_id]) / S[col_id]
        return ret

    def GetDeNormalizedValue(self, col_id, val):
        if self.dataset_name == 'power':
            return val
        U = torch.Tensor(Norm_us[self.dataset_name])
        S = torch.Tensor(Norm_ss[self.dataset_name])
        ret = val * S[col_id] + U[col_id]
        return ret

    def GetLegalRange(self, col, op, val):
        """ legal range for a column """
        col_id = self.getColID(col)
        add_one = self.delta[col_id]
        if op == '=':
            l = self.GetNormalizedValue(col_id, val)
            r = self.GetNormalizedValue(col_id, val + add_one)
        elif op == '>' or op == '>=':
            l = self.GetNormalizedValue(col_id, val)
            r = self.GetNormalizedValue(col_id, self.Maxs[col_id] + add_one)
        elif op == '<' or op == '<=':
            l = self.GetNormalizedValue(col_id, self.Mins[col_id])
            r = self.GetNormalizedValue(col_id, val + add_one)
        elif op == 'in':
            l = self.GetNormalizedValue(col_id, val[0])
            r = self.GetNormalizedValue(col_id, val[1] + add_one)

        return [l, r]

    def getLegalRangeQuery(self, query):
        """legal range for a query"""
        cols, ops, vals = query
        cols, ops, vals = completeColumns(self, cols, ops, vals)

        legal_list = [[0., 1.]] * len(vals)
        i = 0
        for co, op, val_i in zip(cols, ops, vals):
            col_id = self.getColID(co)
            if val_i is None:
                legal_list[i] = self.GetLegalRange(co, 'in', [self.Mins[col_id], self.Maxs[col_id]])
            else:
                legal_list[i] = self.GetLegalRange(co, op, val_i)
            i = i + 1
        return legal_list

    def getLegalRangeNQuery(self, queries):
        """ legal ranges for N queries"""
        legal_lists = []

        for query in queries:
            legal_lists.append(self.getLegalRangeQuery(query))
        return legal_lists

    def getLegalRowBools(self, query):
        """ get legal rows (in bool for row id) for a query """
        columns, operators, vals = query
        assert len(columns) == len(operators) == len(vals)

        bools = None
        for c, o, v in zip(columns, operators, vals):
            c = self.data[c]
            if o in OPS.keys():
                inds = OPS[o](c, v)
            else:
                if o == 'in' or o == 'IN':
                    inds = np.greater_equal(c, v[0])
                    inds &= np.less_equal(c, v[1])

            if bools is None:
                bools = inds
            else:
                bools &= inds
        return bools

    def getOracle(self, query):
        """ get oracle result for a COUNT query """
        bools = self.getLegalRowBools(query)
        c = bools.sum()
        return c

    def getOracleForSum(self, query):
        """ get oracle 【sum】 result for a query """
        bools = self.getLegalRowBools(query)
        rows = self.data[bools]
        # todo: expand to others
        return np.sum(rows.values[:, aggCol])

    def getOracleForAverage(self, query):
        """ get oracle 【average】 result for a query """
        bools = self.getLegalRowBools(query)
        rows = self.data[bools]
        # todo: expand to others
        return np.average(rows.values[:, aggCol])

    def getOracleForVariance(self, query):
        """ get oracle 【average】 result for a query """
        bools = self.getLegalRowBools(query)
        rows = self.data[bools]
        # todo: expand to others
        return np.var(rows.values[:, aggCol])

    def getOracleForPercentile(self, query):
        """ get oracle 【percentile】 result for a query """
        bools = self.getLegalRowBools(query)
        rows = self.data[bools]
        return np.percentile(rows.values[:, aggCol], PERCENT)

    def getOracleForMode(self, query):
        """ get oracle 【mode】 result for a query """
        bools = self.getLegalRowBools(query)
        rows = self.data[bools]
        mode = st.mode(rows.values[:, aggCol])
        print("see mode", mode)
        return mode[0]

    def getOracleForRange(self, query):
        """ get oracle 【range】 result for a query """
        bools = self.getLegalRowBools(query)
        rows = self.data[bools]
        # print("oracle result:")
        # print(" -- min")
        # print(np.min(rows.values[:, aggCol]))
        # print(" -- max")
        # print(np.max(rows.values[:, aggCol]))
        # print(" -- range")
        # print(np.max(rows.values[:, aggCol]) - np.min(rows.values[:, aggCol]))
        return np.max(rows.values[:, aggCol]) - np.min(rows.values[:, aggCol])

    def getAggOracle(self, query, agg_type='count'):
        if agg_type == 'count':
            return self.getOracle(query)
        elif agg_type == 'sum':
            print("Get oracle of agg for 【sum】!")
            return self.getOracleForSum(query)
        elif agg_type == 'average':
            print("Get oracle of agg for 【average】!")
            return self.getOracleForAverage(query)
        elif agg_type == 'variance':
            print("Get oracle of agg for 【variance】!")
            return self.getOracleForVariance(query)
        elif agg_type == 'percentile':
            print("Get oracle of agg for 【percentile】!")
            return self.getOracleForPercentile(query)
        elif agg_type == 'mode':
            print("Get oracle of agg for 【mode】!")
            return self.getOracleForMode(query)
        elif agg_type == 'range':
            print("Get oracle of agg for 【percentile】!")
            return self.getOracleForRange(query)
    def getAndSaveOracle(self, queries, querySeed=2345, agg_type=agg_type):
        """ Calculate oracle results for input queries and save the results"""
        oracle_cards = self.getAllOracle(queries, query_seed, agg_type=agg_type)
        df = pd.DataFrame(oracle_cards, columns=['true_card'])

        """ Change it to your own path """
        print("Save oracle results to :")
        print(PROJECT_PATH + 'evaluate/oracle/{}_rng-{}.csv'.format(self.dataset_name, querySeed))
        df.to_csv(PROJECT_PATH + 'evaluate/oracle/{}_rng-{}.csv'.format(self.dataset_name, querySeed),
                  index=False)
        return


    def getAllOracle(self, queries, agg_type=agg_type, querySeed=2345):
        """ Calculate oracle results for input queries """
        n = len(queries)
        oracle_cards = np.empty(n)
        for i, query in enumerate(queries):
            oracle_cards[i] = self.getAggOracle(query, agg_type=agg_type)
        #             oracle_cards[i] = self.getOracle(query)
        if agg_type == 'count':
            oracle_cards = oracle_cards.astype(np.int)
        return oracle_cards

    def getColumnEnc(self, query):
        cols, ops, vals = query
        ret = torch.zeros(self.dim, dtype=torch.int)
        for col in cols:
            col_id = self.getColID(col)
            ret[col_id] = 1
        return ret


    def getNColumnEnc(self, queries):
        ret = []
        for query in queries:
            ret.append(self.getColumnEnc(query))
        ret = torch.stack(ret, dim=0)
        return ret