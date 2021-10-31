
import datetime as dt
import numpy as np
import pandas as pd
from util import get_data, plot_data

def compute_portvals(df_order, sd, ed, start_val=100000, commission=9.95, impact=0.005):

    df = df_order
    df.sort_index(ascending=True, inplace=True)
    start_date = df.index.min()
    end_date = df.index.max()
    dates = pd.date_range(start_date,end_date)
    symbols = df.columns[-1]
    commission = commission
    impact = impact
    #print(symbols)



    symbol_price = get_data([symbols], pd.date_range(start_date, end_date))
    symbol_price.fillna(method="ffill", inplace=True)
    symbol_price.fillna(method="bfill", inplace=True)
    #symbol_price_SPY = symbol_price[['SPY']]
    symbol_price = symbol_price.drop(['SPY'], axis=1)

    symbol_price['cash'] = np.ones(symbol_price.shape[0])
    #print(symbol_price)
    trades = symbol_price.copy()
    trades[:] = 0
    #print(df)
    #print(trades)

# modified for project 6
    for index, row in df.iterrows():

        if index not in pd.date_range(start_date, end_date):
            continue
        if row[symbols] > 0:
            trades.ix[index, symbols] += row[symbols]
            money = commission + symbol_price.ix[index, symbols] * (1 + impact) * row[symbols]
            trades.ix[index, 'cash'] -= money
        else:
            trades.ix[index, symbols] += row[symbols]
            money = symbol_price.ix[index, symbols] * (1 - impact) * row[symbols] + commission
            trades.ix[index, 'cash'] -= money
    #print(trades)

    holds = trades.copy()
    holds.ix[start_date, 'cash'] = holds.ix[start_date, 'cash'] + start_val
    holds = holds.cumsum(axis=0)
    #print(holds)
    #print (symbol_price)

    # values
    values = symbol_price * holds
    portvals =values.sum(axis=1)

    daily_return =portvals.copy()
    daily_return[1:] = ((portvals/portvals.shift(1)) - 1).iloc[1:]
    daily_return.iloc[0] = 0
    daily_return = daily_return[1:]
    #print(daily_return)

    cr = (portvals[-1]/portvals[0]) - 1
    #print(cr)
    adr = daily_return.mean()
    stdr = daily_return.std()
    return portvals, cr, adr, stdr


def author():
    return 'yliu3306'

