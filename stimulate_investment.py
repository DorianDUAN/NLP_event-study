import pandas as pd
import numpy as np
import datetime as dt
from sklearn import linear_model
import scipy.stats as st
import yfinance as yf

def pos_investment(
        data_ret,
        eventdate,
        ticker,
        estimation_period=252,
        before_event=20,
        event_window_start=-20,
        event_window_end=20,
        benchmark="^GSPC",
):
    """
    Function takes in the historical returns, an event date of a stock, returns the expected average return
    if we invest in AAPL on the announcement date.
    holding for 5/10/20 days.
    如何动态调整时间和投资量??
    """

    # Generate post-event indicator
    data_ret["post_event"] = (data_ret["Date"] >= eventdate).astype(
        int
    )  # 1 if after event, 0 otherwise
    data_ret = (
        data_ret.reset_index()
    )  # pushes out the current index column and create a new one

    # Identify the index for the event date
    event_date_index = data_ret.groupby(["post_event"])["index"].transform("min").max()
    data_ret["event_date_index"] = event_date_index

    # Create the variable day relative to event
    data_ret["rel_day"] = data_ret["index"] - data_ret["event_date_index"]

    # Identify estimation period
    estimation = data_ret[
        (data_ret["rel_day"] < -before_event)
        & (data_ret["rel_day"] >= -estimation_period - before_event)
        ]

    # Identify event period
    event = data_ret[
        (data_ret["rel_day"] <= event_window_end)
        & (data_ret["rel_day"] >= event_window_start)
        ]

    # Calculate expected returns with the market model
    x_df = estimation[benchmark].values.reshape(-1, 1)

    # Create an empty list to store betas
    betas = []

    # Calculate betas for the market model
    for y in [benchmark, ticker]:
        y_df = estimation[y].values.reshape(-1, 1)
        reg = linear_model.LinearRegression()
        betas.append(reg.fit(x_df, y_df).coef_)

    # Convert the list to a Numpy Array
    beta_np = np.array(betas)
    beta_np

    # Expected Returns via Beta
    # Need Numpy Array to do Calculations!
    sp500array = event[benchmark].values
    expected_returns = np.outer(sp500array, beta_np)
    expected_returns = pd.DataFrame(expected_returns, index=event.index)
    expected_returns.columns = [benchmark, ticker]
    expected_returns = expected_returns.rename(columns={ticker: "expected_return"})
    del expected_returns[benchmark]

    # Abnormal Returns
    event = pd.concat([event, expected_returns], axis=1, ignore_index=False)

    event["abnormal_return"] = event[ticker] - event["expected_return"]
    event[ticker] = event[ticker]+1
    event["expected_return"] = event["expected_return"]+1
    event[benchmark] = event[benchmark] + 1

    # Event Holding period return of AAPL
    winar1 = event[(event["rel_day"] <= 5) & (event["rel_day"] >= 0)][
        ticker
    ].product()  # HPR[0,5]
    winar2 = event[(event["rel_day"] <= 10) & (event["rel_day"] >= 0)][
        ticker
    ].product()  # HPR[0,10]
    winar3 = event[(event["rel_day"] <= 20) & (event["rel_day"] >= 0)][
        ticker
    ].product()  # HPR[0,20]

    # Expected Holding period return without the event
    winar4 = event[(event["rel_day"] <= 5) & (event["rel_day"] >= 0)][
        "expected_return"
    ].product()  # HPR[0,5]
    winar5 = event[(event["rel_day"] <= 10) & (event["rel_day"] >= 0)][
        "expected_return"
    ].product()  # HPR[0,10]
    winar6 = event[(event["rel_day"] <= 20) & (event["rel_day"] >= 0)][
        "expected_return"
    ].product()  # HPR[0,20]

    # Expected Holding period return without the event
    winar7 = event[(event["rel_day"] <= 5) & (event["rel_day"] >= 0)][
        "expected_return"
    ].product()  # HPR[0,5]
    winar8 = event[(event["rel_day"] <= 10) & (event["rel_day"] >= 0)][
        "expected_return"
    ].product()  # HPR[0,10]
    winar9 = event[(event["rel_day"] <= 20) & (event["rel_day"] >= 0)][
        "expected_return"
    ].product()  # HPR[0,20]

    # Event CAR
    winar7 = event[(event["rel_day"] <= 5) & (event["rel_day"] >= 0)][
        "abnormal_return"
    ].sum()  # CAR[0,5]
    winar8 = event[(event["rel_day"] <= 10) & (event["rel_day"] >= 0)][
        "abnormal_return"
    ].sum()  # CAR[0,10]
    winar9 = event[(event["rel_day"] <= 20) & (event["rel_day"] >= 0)][
        "abnormal_return"
    ].sum()  # CAR[0,20]

    market_port1 = event[(event["rel_day"] <= 5) & (event["rel_day"] >= 0)][
        benchmark
    ].product()
    market_port2 = event[(event["rel_day"] <= 10) & (event["rel_day"] >= 0)][
        benchmark
    ].product()
    market_port3 = event[(event["rel_day"] <= 20) & (event["rel_day"] >= 0)][
        benchmark
    ].product()

    return (
        winar1,
        winar2,
        winar3,
        winar4,
        winar5,
        winar6,
        market_port1,
        market_port2,
        market_port3,
    )

# read dataset for AAPL and S&P from yfinance from start to end
symbols_list = ["^GSPC", "AAPL",]
start = dt.datetime(2015, 1, 1)
end = dt.datetime(2020, 12, 31)

data = yf.download(symbols_list, start=start, end=end)
# Calculate returns
main_data = data["Adj Close"] / data["Adj Close"].shift(1) - 1
main_data = main_data.dropna()
main_data = main_data.reset_index()

# define the event, and the firm's name is stored in ticker
benchmark = "^GSPC"
ticker = "AAPL"
eventdate = dt.datetime(2020, 7, 30)

hprs = pos_investment(main_data, eventdate, "AAPL")

