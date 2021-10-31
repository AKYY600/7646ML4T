""""""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Atlanta, Georgia 30332  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
All Rights Reserved  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Template code for CS 4646/7646  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
or edited.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT honor code violation.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
-----do not edit anything above this line---  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Student Name: Tucker Balch (replace with your name)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT User ID: tb34 (replace with your User ID)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT ID: 900897987 (replace with your GT ID)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import datetime as dt  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import random
import numpy as np
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import pandas as pd  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import util as ut
import BagLearner as bl
import indicators as ind

from marketsimcode import compute_portvals
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
class StrategyLearner(object):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        If verbose = False your code should not generate ANY output.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type verbose: bool  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type impact: float  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param commission: The commission amount charged, defaults to 0.0  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type commission: float  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # constructor  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        Constructor method  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        self.verbose = verbose  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        self.impact = impact  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        self.commission = commission
        self.returndays = 10
        self.learner = bl.BagLearner(kwargs = {"leaf_size":5,"verbose":False},bags = 20)

  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # this method should create a QLearner, and train it for trading  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			

    def testdata(self, syms, sd, ed, prices):

        mean, upper, lower = ind.Bollinger(sd, ed, syms, False)
        sma, p_s = ind.SMA(sd, ed, syms, False)
        momentum = ind.momentum(sd, ed, syms, False)

        x_data = np.zeros((len(prices) - self.returndays, 5))
        for i in range(len(prices) - self.returndays):

            x_data[i][0] = (upper.iloc[i] - lower.iloc[i]) / 4
            x_data[i][1] = prices.iloc[i] - upper.iloc[i]
            x_data[i][2] = prices.iloc[i] - lower.iloc[i]
            x_data[i][3] = p_s.iloc[i]
            x_data[i][4] = momentum.iloc[i]
        return x_data

    def add_evidence(
        self,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        symbol="IBM",
        sd=dt.datetime(2008, 1, 1),  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        ed=dt.datetime(2009, 1, 1),  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        sv=10000,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    ):

        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
        if self.verbose:
            print(prices)
        mean, upper, lower = ind.Bollinger(sd, ed, symbol, False)
        sma, p_s = ind.SMA(sd, ed, symbol, False)
        momentum = ind.momentum(sd, ed, symbol, False)

        x_data = np.zeros((len(prices) - self.returndays, 7))

        for i in range(len(prices) - self.returndays):
            x_data[i][0] = prices.iloc[i]
            x_data[i][1] = prices.iloc[i + self.returndays]
            x_data[i][2] = (upper.iloc[i] - lower.iloc[i]) / 4
            x_data[i][3] = prices.iloc[i] - upper.iloc[i]
            x_data[i][4] = prices.iloc[i] - lower.iloc[i]
            x_data[i][5] = p_s.iloc[i]
            x_data[i][6] = momentum.iloc[i]
        x_train = x_data[:,2:]
        y_train = []

        for i in range(x_data.shape[0]):
            if x_data[i,1]/x_data[i,0] > 1.01+self.impact:
                y_train.append(1)
            elif x_data[i,1]/x_data[i,0] < 0.99+self.impact:
                y_train.append(-1)
            else:
                y_train.append(0)

        y_train = np.array(y_train)
        #print(y_train)
        x_train = x_train[20:,:]
        y_train = y_train[20:]


        self.learner.add_evidence(x_train, y_train)

        #print(y_train)

    def testPolicy(
            self,
            symbol="IBM",
            sd=dt.datetime(2009, 1, 1),
            ed=dt.datetime(2010, 1, 1),
            sv=10000,
    ):

        syms = symbol
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        prices = prices_all[[symbol, ]]  # only portfolio symbols
        prices_SPY = prices_all["SPY"] # only SPY, for comparison later

        df_trades = prices_all[['SPY']]
        df_trades = df_trades.rename(columns={'SPY': syms}).astype({syms: 'int32'})
        df_trades[:] = 0
        Date = df_trades.index
        current_share = 0

        x_test = self.testdata(symbol, sd, ed, prices)
        y_test = self.learner.query(x_test)

        for i in range(len(Date) - self.returndays):
            if y_test[i] == 1:
                action = 1000 - current_share

            elif y_test[i] == -1:
                action = -1000 - current_share

            else:
                action = 0

            df_trades.loc[Date[i]].loc[syms] = action
            current_share += action

        #print (df_trades)
        return df_trades



if __name__ == "__main__":
    print("One does not simply think up a strategy")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			




