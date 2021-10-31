
import datetime as dt
import numpy as np
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt

def author():
    return 'yliu3306'

#Bollinger Bands
def Bollinger(sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sym=['JPM'],gen_plot=True):
    # get data
    dates = pd.date_range(sd,ed)
    symbol = sym
    price = get_data([symbol],dates)
    price.fillna(method="ffill", inplace=True)
    price.fillna(method="bfill", inplace=True)
    price = price[symbol]
    price = price/price.ix[0,]

    std = price.rolling(window=20).std()
    df = price.rolling(window=20).mean()
    upper = df + 2*std
    lower = df - 2*std

    if gen_plot:
        ax = df.plot(title = "Bollinger Band",label='SMA')
        upper.plot(label='Upper Band', ax=ax)
        lower.plot(label='Lower Band', ax=ax)
        price.plot(label='JPM Price', ax=ax)
        ax.legend(["SMA", "Upper Band", "Lower Band", "JPM Price"]);
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='lower right')
        plt.savefig('Bollinger Band.png')
        plt.clf()

    return df, upper, lower

#SMA
def SMA(sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sym=['JPM'],gen_plot=True):
    dates = pd.date_range(sd, ed)
    symbol = sym
    price = get_data([symbol], dates)
    price.fillna(method="ffill", inplace=True)
    price.fillna(method="bfill", inplace=True)
    price = price[symbol]
    price = price / price.ix[0,]

    sma = price.rolling(window=20).mean()
    p_s = price.divide(sma, axis = 'index')
    p_s.columns = ['Price/SMA']
    sma = sma.iloc[20:]

    if gen_plot:
        ax = sma.plot(title = "SMA", label = 'SMA')
        p_s.plot(label = 'Price/SMA', ax = ax)
        price.plot(label = 'Price', ax = ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='lower right')
        plt.savefig('SMA.png')
        plt.clf()

    return sma, p_s

# momentum
def momentum(sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sym=['JPM'],gen_plot=True):
    dates = pd.date_range(sd, ed)
    symbol = sym
    price = get_data([symbol], dates)
    price.fillna(method="ffill", inplace=True)
    price.fillna(method="bfill", inplace=True)
    price = price[symbol]
    price = price / price.ix[0,]

    momentum = pd.DataFrame(data=0,index=price.index, columns = ['Momentum'])
    momentum.ix[10:]=price/price.shift(periods = 10) -1

    if gen_plot:
        ax = price.plot(title = "Momentum", label='JPM Price')
        momentum.plot(label = 'Momentum', ax = ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='lower right')
        plt.savefig('Momentum.png')
        plt.clf()

    return momentum

# volatility
def STD(sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sym=['JPM'],gen_plot=True):
    dates = pd.date_range(sd, ed)
    symbol = sym
    price = get_data([symbol], dates)
    price.fillna(method="ffill", inplace=True)
    price.fillna(method="bfill", inplace=True)
    price = price[symbol]
    price = price / price.ix[0,]

    std = price.rolling(window=20).std()

    if gen_plot:
        ax = std.plot(title="Standard Deviation", label='STD')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='lower right')
        plt.savefig('STD.png')
        plt.clf()

    return std


# EMA
def EMA(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sym=['JPM'], gen_plot=True):
    dates = pd.date_range(sd, ed)
    symbol = sym
    price = get_data([symbol], dates)
    price.fillna(method="ffill", inplace=True)
    price.fillna(method="bfill", inplace=True)
    price = price[symbol]
    price = price / price.ix[0,]

    ema = price.rolling(window=20).mean()
    multiplier = 2/ (20 +1)
    ema = price * multiplier + ema.shift(periods = 1) * (1-multiplier)
    ema = ema.iloc[21:]

    if gen_plot:
        ax = ema.plot(title="EMA", label='EMA')
        price.plot(label='Price', ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='lower right')
        plt.savefig('EMA.png')
        plt.clf()

    return ema

if __name__=="__main__":
    Bollinger(dt.datetime(2008,1,1),dt.datetime(2009,12,31), "JPM", True)
    SMA(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), "JPM", True)
    momentum(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), "JPM", True)
    STD(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), "JPM", True)
    EMA(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), "JPM", True)


