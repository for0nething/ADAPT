import numpy as np


COLS = {
    'power': ['c0', 'c1', 'c2', 'c3', 'c4', 'c5'],
    'BJAQ' : ['PM2.5', 'PM10', 'NO2', 'O3', 'TEMP'],
    'flights': ['MONTH', 'DAY',  'AIRLINE', 'FLIGHT_NUMBER',
                'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE',
                'SCHEDULED_ARRIVAL'],
    'imdbfull': ['movie_id', 'company_id', 'company_type_id', 'kind_id', 'production_year', 'country_code', 'info_type_id']
}

filterNum = {
    'power': (1, 3),
    'BJAQ': (1, 2),
    'flights': (2, 5),
    'imdbfull': (1, 4)

    # 'power': (3, 6),
    # 'BJAQ': (3, 5),
    # 'flights': (3, 7),
    # 'imdbfull': (3, 6)
}

sensible = {
    'power': np.ones(6),
    'BJAQ': np.ones(5),
    'flights': np.ones(8),
    'imdbfull': np.ones(7)
}

deltas = {
    'power': np.array([0, 0, 0, 0, 0, 0]),
    'BJAQ': np.array([1, 1, 1, 1, 0.1]),
    'flights': np.ones(8),
    'imdbfull': np.ones(7)
}

Norm_us = {
    'BJAQ': np.array([79.9326, 105.07354, 51.070656, 57.876205, 13.568575]),
    'flights': np.array([7.0240955, 16.204762, 7.1802936, 2491.3682, 112.12262, 90.10231, 1330.1025, 1494.3085]),
    'imdbfull': np.array([1663234.9, 27729.5, 0.91545355, 2.1881678, 1987.9879, 9.050387, 1.504228])
}

Norm_ss = {
    'BJAQ': np.array([80.15541, 91.38018, 35.06305, 56.71038, 11.425453]),
    'flights': np.array([3.4173381, 8.788036, 4.0915422, 1690.3027, 172.18059, 160.388, 483.7519, 507.16467]),
    'imdbfull': np.array([663584.44, 44231.875, 0.5710898, 1.3034894, 29.577124, 14.158293, 0.8702347])
}


OPS = {
    '>':np.greater,
    '<':np.less,
    '>=':np.greater_equal,
    '<=':np.less_equal,
    '=':np.equal,
}
