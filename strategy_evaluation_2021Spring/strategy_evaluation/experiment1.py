import datetime as dt
import random
import numpy as np

import pandas as pd
import util as ut
import BagLearner as bl
import indicators as ind
import StrategyLearner as sl
import ManualStrategy as ms
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals

def author():
    return 'yliu3306'

def experiment1():
    sv = 100000
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    commission = 9.95
    impact = 0.005

    # Strategy Learner
    strategy = sl.StrategyLearner(verbose=False, impact=0.005, commission=9.95)
    strategy.add_evidence(symbol, sd, ed, sv)
    strategy.testPolicy(symbol, sd, ed, sv)

    learnertrade = strategy.testPolicy(symbol, sd, ed, sv)
    learnerportvals, learnercr, learnermean, learnerstd = compute_portvals(learnertrade, sd, ed,
                                                                           sv, commission, impact)
    # Bench
    bench_trade = ms.benchmark(sd, ed, symbol, sv)
    benchportvals, benchcr, benchmean, benchstd = compute_portvals(bench_trade, sd, ed, sv, commission, impact)

    #manual
    manualtrade = ms.testPolicy(sd, ed, symbol, sv)
    manualportvals, manualcr, manualmean, manualstd = compute_portvals(manualtrade, sd, ed, sv, commission, impact)
    #print(manualportvals)

    print("StrategyLearner CR:", learnercr)
    print("StrategyLearner Mean:", learnermean)
    print("StrategyLearner STD:", learnerstd)
    print("Benchmark CR:", benchcr)
    print("Benchmark Mean:", benchmean)
    print("Benchmark STD:", benchstd)
    print("Manual CR:", manualcr)
    print("Manual Mean:", manualmean)
    print("Manual STD:", manualstd)

    #plot

    Normed_bench = benchportvals / benchportvals.iloc[0]
    Normed_manual = manualportvals / manualportvals.iloc[0]
    Normed_strategy = learnerportvals / learnerportvals.iloc[0]
    #print (Normed_strategy)

    plt.title("Experiment1")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio")
    #plt.grid()
    plt.plot(Normed_bench, label="benchmark", color="green")
    plt.plot(Normed_manual, label="manual strategy", color="red")
    plt.plot(Normed_strategy, label="strategy strategy", color="blue")

    plt.legend()
    plt.savefig('experiment1.png')
    plt.clf()


if __name__ == "__main__":
    sv = 100000
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    commission = 9.95
    impact = 0.005

    experiment1()