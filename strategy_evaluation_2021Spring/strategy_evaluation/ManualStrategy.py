import datetime as dt
import numpy as np
import pandas as pd
from util import get_data, plot_data
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
import indicators as ind

def author():
    return 'yliu3306'

def testPolicy(sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sym=['JPM'], sv=100000):

    symbols = sym
    df = get_data([symbols], pd.date_range(sd, ed))
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    price = df[symbols]
    df_trades = df[['SPY']]
    df_trades = df_trades.rename(columns={'SPY': symbols}).astype({symbols: 'int32'})
    df_trades[:] = 0
    Date = df_trades.index
    current_share = 0

    mean, upper, lower = ind.Bollinger(sd, ed, sym, False)
    sma, p_s = ind.SMA(sd, ed, sym, False)
    momentum = ind.momentum(sd, ed, sym, False)
    price_BB = price / price.ix[0,]
    BB = (price_BB-mean)/((upper-lower)/2)
    #print(momentum)
    #print(p_s)

    for i in range(len(Date) - 1):
        if (BB[i]<-0.7 and momentum[i]<-0.02) or (p_s[i]<0.95 and momentum[i]<-0.02):
            action = 1000 - current_share

        elif (BB[i]>0.7 and momentum[i]>0.02) or (p_s[i]>1.05 and momentum[i]>0.02):
            action = -1000 - current_share

        else:
            action = 0

        df_trades.loc[Date[i]].loc[symbols] = action
        current_share += action
    #print (df_trades)
    return df_trades


def benchmark(sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sym=['JPM'], sv=100000):
    symbol = sym
    bench_price = get_data([symbol], pd.date_range(sd, ed))
    bench_price.fillna(method="ffill", inplace=True)
    bench_price.fillna(method="bfill", inplace=True)
    bench_price = bench_price[symbol]
    bench_trade = bench_price.copy()
    bench_trade[:] = 0
    bench_trade.iloc[0] = 1000
    bench_trade = pd.DataFrame(data=bench_trade, index=bench_price.index, columns=['JPM'])
    # print(bench_trade)
    return bench_trade


if __name__ == "__main__":
    start_val = 100000
    symbol = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    # print(start_date.date())
    # print(end_date)
    bench_trade = benchmark(dt.datetime(2008,1,1), dt.datetime(2009,12,31), "JPM", 100000)

    benchportvals, benchcr, benchmean, benchstd = compute_portvals(bench_trade, start_date, end_date, start_val, 0, 0)

    opttrade = testPolicy(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), "JPM", 100000)
    # print(opttrade)
    optportvals, optcr, optmean, optstd = compute_portvals(opttrade, start_date, end_date, start_val, 0, 0)

    #print (optportvals)

    print("In Sample Benchmark CR:", benchcr)
    print("In Sample Benchmark Mean:", benchmean)
    print("In Sample Benchmark STD:", benchstd)

    print("In Sample Optimal CR:", optcr)
    print("In Sample Optimal Mean:", optmean)
    print("In Sample Optimal STD:", optstd)

    Normed_bench = benchportvals / benchportvals.iloc[0]
    Normed_opt = optportvals / optportvals.iloc[0]
    #print (Normed_opt)

    plt.title("In Sample Manual Strategy")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio")
    #plt.grid()
    plt.plot(Normed_bench, label="benchmark", color="green")
    plt.plot(Normed_opt, label="manual strategy", color="red")

    Date = opttrade.index
    symbols =opttrade.columns
    #print (opttrade.loc[Date[5]],symbols)

    for i in range(len(Date) - 1):
        #print(opttrade.loc[Date[i], symbols][0])

        if opttrade.loc[Date[i], symbols][0] > 0:
            plt.axvline(x=Date[i], ymin=0, ymax=1.5,color="blue")
            #print('LONG')
        elif opttrade.loc[Date[i], symbols][0] < 0:
            plt.axvline(x=Date[i], ymin=0, ymax=1.5, color="black")
            #print('SHORT')


    plt.legend()
    plt.savefig('insamplemanual.png')
    plt.clf()

    #outsample

    start_val = 100000
    symbol = "JPM"
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2011, 12, 31)
    # print(start_date.date())
    # print(end_date)
    bench_trade = benchmark(dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), "JPM", 100000)

    benchportvals, benchcr, benchmean, benchstd = compute_portvals(bench_trade, start_date, end_date, start_val, 0, 0)

    opttrade = testPolicy(dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), "JPM", 100000)
    # print(opttrade)
    optportvals, optcr, optmean, optstd = compute_portvals(opttrade, start_date, end_date, start_val, 0, 0)

    # print (optportvals)

    print("Out-Sample Benchmark CR:", benchcr)
    print("Out-Sample Benchmark Mean:", benchmean)
    print("Out-Sample Benchmark STD:", benchstd)

    print("Out-Sample Manual CR:", optcr)
    print("Out-Sample Manual Mean:", optmean)
    print("Out-Sample Manual STD:", optstd)

    Normed_bench = benchportvals / benchportvals.iloc[0]
    Normed_opt = optportvals / optportvals.iloc[0]
    # print (Normed_opt)

    plt.title("Out-sample Manual Strategy")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio")
    # plt.grid()
    plt.plot(Normed_bench, label="benchmark", color="green")
    plt.plot(Normed_opt, label="manual strategy", color="red")

    Date = opttrade.index
    symbols = opttrade.columns
    # print (opttrade.loc[Date[5]],symbols)

    for i in range(len(Date) - 1):
        # print(opttrade.loc[Date[i], symbols][0])

        if opttrade.loc[Date[i], symbols][0] > 0:
            plt.axvline(x=Date[i], ymin=0, ymax=1.5, color="blue")
            # print('LONG')
        elif opttrade.loc[Date[i], symbols][0] < 0:
            plt.axvline(x=Date[i], ymin=0, ymax=1.5, color="black")
            # print('SHORT')

    plt.legend()
    plt.savefig('outsamplemanual.png')
    plt.clf()




