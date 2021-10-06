import pandas as pd
import numpy as np
import datetime as dt
from sklearn import linear_model
import scipy.stats as st
import yfinance as yf
from IPython.display import display

from event_study import do_event_study
from stimulate_investment import pos_investment

def analysis(hprs, index1, index2):
    significant = hprs[hprs['P-Value'+index1] < 0.05]
    insignificant = hprs[hprs['P-Value'+index1] >= 0.05]
    return (
        significant["winar"+index1].mean(),
        insignificant["winar"+index1].mean(),
        len(significant)/len(hprs),
        1-(len(significant) / len(hprs)),
        hprs["winar"+index2].mean()
    )

if __name__ == '__main__':
    # read dataset for six stocks from yfinance from start to end
    symbols_list = ["^GSPC", "AAPL"]
    start = dt.datetime(2005, 1, 1)
    end = dt.datetime(2021, 10, 1)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_row", None)
    pd.set_option("display.max_columns", None)

    data = yf.download(symbols_list, start=start, end=end)
    # Calculate returns
    main_data = data["Adj Close"] / data["Adj Close"].shift(1) - 1
    main_data = main_data.dropna()
    main_data = main_data.reset_index()

    # define the event, and the firm's name is stored in ticker
    benchmark = "^GSPC"
    ticker = "AAPL"
    sentiment = -1 # need to retrive from NLP
    # define the event study window.
    estimation_period = 252
    before_event = 20
    event_window_start = -20
    event_window_end = 20

    # event list
    data_events = pd.read_csv(
        "dataset/AAPL Earnings Surprise.csv", na_values=["."], parse_dates=["Date"]
    )
    pos_events = data_events[data_events["Type"] == sentiment]
    del pos_events["Type"]

    hprs = []
    for index, row in pos_events.iterrows():
        data_ret = main_data[["Date", ticker, benchmark]].copy()
        hprs.append(pos_investment(data_ret, ticker=ticker, eventdate=row["Date"]))

    hprs = pd.DataFrame(hprs)
    hprs.columns = [
        "winar1",
        "winar2",
        "winar3",
        "winar4",
        "winar5",
        "winar6",
        "market_port1",
        "market_port2",
        "market_port3",
    ]
    # Calculate the excess HPR off the expected HPR using beta of AAPL
    hprs["excess1"] = hprs["winar4"] - hprs["winar1"]
    hprs["excess2"] = hprs["winar5"] - hprs["winar2"]
    hprs["excess3"] = hprs["winar6"] - hprs["winar3"]

    # Calculate the Mean and Standard Deviation of the AAR
    mean_AAR = hprs.mean()
    std_AAR = hprs.sem()

    hprs['T-Test1'] = hprs["excess1"] / std_AAR["excess1"]
    hprs['T-Test2'] = hprs["excess2"] / std_AAR["excess2"]
    hprs['T-Test3'] = hprs["excess3"] / std_AAR["excess3"]
    # test if excess hpr is significant
    hprs['P-Value1'] = st.t.sf(np.abs(hprs['T-Test1']), len(hprs) - 1) * 2
    hprs['P-Value2'] = st.t.sf(np.abs(hprs['T-Test2']), len(hprs) - 1) * 2
    hprs['P-Value3'] = st.t.sf(np.abs(hprs['T-Test3']), len(hprs) - 1) * 2

    # Display is a great method to show multiple outputs at once
    # display(hprs.info())
    stats = []
    stats.append(analysis(hprs, "1", "4"))
    stats.append(analysis(hprs, "2", "5"))
    stats.append(analysis(hprs, "3", "6"))
    stats = pd.DataFrame(stats)
    stats.columns = [
        "Avg. HPR of underreaction",
        "Avg. HPR of efficient cases",
        "Prob. of underreaction",
        "Prob. of efficient case",
        "Exp. HPR without event",
    ]
    stats = stats.fillna(1)
    stats["Expected Return"]=(
            stats["Avg. HPR of underreaction"] * stats["Prob. of underreaction"] +
            stats["Avg. HPR of efficient cases"] * stats["Prob. of efficient case"]
    )

    print(stats)



