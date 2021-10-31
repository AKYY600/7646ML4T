""""""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""MC1-P2: Optimize a portfolio.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
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
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Student Name: Yuanyuan Liu (replace with your name)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT User ID: yliu3306 (replace with your User ID)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT ID: 903560210 (replace with your GT ID)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import datetime as dt  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import numpy as np  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import matplotlib.pyplot as plt  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import pandas as pd  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
from util import get_data, plot_data
import scipy.optimize as sp

  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
# This is the function that will be tested by the autograder  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
# The student must update this code to properly implement the functionality  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    statistics.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type sd: datetime  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type ed: datetime  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        symbol in the data directory)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type syms: list  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        code with gen_plot = False.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type gen_plot: bool  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        standard deviation of daily returns, and Sharpe ratio  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :rtype: tuple  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # Read in adjusted closing prices for given symbols, date range  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    dates = pd.date_range(sd, ed)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    prices_all = get_data(syms, dates)# automatically adds SPY
    prices_all.fillna(method="ffill", inplace=True)
    prices_all.fillna(method="bfill", inplace=True)
    prices = prices_all[syms]# only portfolio symbols
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
    print (prices)
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # find the allocations for the optimal portfolio
    numb_stocks = len(syms)
    initial_guess = numb_stocks*[1.0/numb_stocks]
    bnds = tuple((0,1) for x in range(numb_stocks))
    cons = {'type': 'eq', 'fun': lambda allocs: 1.0 - np.sum(allocs)}
    allocs = sp.minimize(min_sharpe, initial_guess,
                         args=(prices,), bounds=bnds, constraints=cons, method='SLSQP').x
    #print(allocs)
    port_val, cr, adr, sddr, sr = stockcalculation(allocs, prices)
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat(
            [port_val / port_val.ix[0, :], prices_SPY / prices_SPY.ix[0, :]], keys=["Portfolio", "SPY"], axis=1)
        finalfigure = df_temp.plot()
        finalfigure.set_title('Daily Portfolio Value vs SPY')
        finalfigure.set_ylabel('Price')
        finalfigure.set_xlabel('Date')
        plt.grid(True)
        plt.savefig('./Optimization.png')

    return allocs, cr, adr, sddr, sr
    # Get daily portfolio value
def stockcalculation(allocs, prices):
    norm_prices = prices/prices.ix[0,:]
    allcs_prices = norm_prices * allocs
    port_val = allcs_prices.sum(axis=1)# add code here to compute daily portfolio values
    daily_rets = port_val.pct_change(1)
    daily_rets = daily_rets[1:]

    cr = port_val[-1]/port_val[0] -1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    #risk free is 0
    sr = (adr - 0) / sddr
    sr = (252 ** 0.5) * sr
    return port_val, cr, adr, sddr, sr

def min_sharpe(allocs, prices):
    return -stockcalculation(allocs, prices)[4]
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
def test_code():  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    This function WILL NOT be called by the auto grader.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # Assess the portfolio  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # Print statistics  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Start Date: {start_date}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"End Date: {end_date}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Symbols: {symbols}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Allocations:{allocations}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Sharpe Ratio: {sr}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Average Daily Return: {adr}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Cumulative Return: {cr}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
if __name__ == "__main__":  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # This code WILL NOT be called by the auto grader  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # Do not assume that it will be called  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    test_code()  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
