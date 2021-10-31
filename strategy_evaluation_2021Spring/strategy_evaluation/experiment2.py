import datetime as dt
import random
import numpy as np

import pandas as pd
import util as ut
import BagLearner as bl
import indicators as ind
import StrategyLearner as sl
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals

def author():
    return 'yliu3306'

def experiment2():
    sv = 100000
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    commission = 0
    # impact = 0
    impact = 0.0
    strategy = sl.StrategyLearner(verbose=False, impact = 0.005, commission=0)
    strategy.add_evidence(symbol, sd, ed, sv)

    learnertrade1 = strategy.testPolicy(symbol, sd, ed, sv)
    learnerportvals1, learnercr1, learnermean1, learnerstd1 = compute_portvals(learnertrade1, sd, ed, commission, impact)

    #impact = 0.05
    impact = 0.05
    strategy = sl.StrategyLearner(verbose=False, impact=0.05, commission=0)
    strategy.add_evidence(symbol, sd, ed, sv)

    learnertrade2 = strategy.testPolicy(symbol, sd, ed, sv)
    learnerportvals2, learnercr2, learnermean2, learnerstd2 = compute_portvals(learnertrade2, sd, ed, commission, impact)

    #impact = 0.5
    impact = 0.5
    strategy = sl.StrategyLearner(verbose=False, impact=0.5, commission=0)
    strategy.add_evidence(symbol, sd, ed, sv)

    learnertrade3 = strategy.testPolicy(symbol, sd, ed, sv)
    learnerportvals3, learnercr3, learnermean3, learnerstd3 = compute_portvals(learnertrade3, sd, ed, commission, impact)

    print("Impact = 0.005: CR:", learnercr1, "Mean:", learnermean1, "STD:",learnerstd1)
    print("Impact = 0.05: CR:", learnercr2, "Mean:", learnermean2, "STD:", learnerstd2)
    print("Impact = 0.5: CR:", learnercr3, "Mean:", learnermean3, "STD:", learnerstd3)

    portchart = pd.concat([learnerportvals1,learnerportvals2,learnerportvals3], axis = 1)
    portchart.columns = ['Impact=0.005', 'Impact=0.05', 'Impact=0.5']
    portchart.plot(title='Impact difference on portfolios', use_index = True, color =['Red','Green','Blue'])
    plt.title("Experiment2")
    plt.xlabel("Date")
    plt.ylabel("Portfolio")
    plt.savefig('experiment2.png')
    plt.clf()



if __name__ == "__main__":

    experiment2()
