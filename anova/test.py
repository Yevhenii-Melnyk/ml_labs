from __future__ import division

import pandas
from scipy.stats import f


def compute(data):
    N = data.size
    C = len(data.columns)

    dfc = C - 1
    dfer = N - C
    dft = N - 1

    cm = data.mean()
    tm = data.sum().sum() / N
    n = data.shape[0]

    SSC = sum((tm - cm) ** 2) * n
    MSC = SSC / dfc

    SSE = ((data - cm) ** 2).sum().sum()
    MSE = SSE / dfer

    SST = ((data - tm) ** 2).sum().sum()

    F = MSC / MSE

    alpha = 0.05
    p_value = 1 - f.cdf(F, dfc, dfer)

    print data
    print
    print pandas.DataFrame({'df': [dfc, dfer, dft],
                            'SS': [SSC, SSE, SST],
                            'MS': [MSC, MSE, ''],
                            'F': [F, '', ''],
                            'p value': [p_value, '', '']},
                           columns=['df', 'SS', 'MS', 'F', 'p value'],
                           index=['between', 'within', 'total'])
    print
    if p_value > alpha:
        print "Reject null hypothesis"
    else:
        print "Accept null hypothesis"
    print '~~~~~~~~~'


for idx in range(1, 8):
    name = 'task%d.txt' % (idx,)
    data = pandas.read_csv(name, header=None)
    compute(data)
