import pandas as pd
import numpy as np
import datetime as dt
from sklearn import linear_model
import scipy.stats as st

def do_event_study(
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
    Function takes in the historical returns, an event date of a stock, returns the cumulative abnormal returns (CARS) over
    a specified timeframe

    Parameters:
        data_ret (pd.DataFrame): A dataframe containing daily returns of stock(s) and the specified benchmark. columns: tickers, rows: returns
        eventdate (datetime): the event date to be studied. eventdate must be within the date frame of data_ret
        ticker (str): ticker or CUSIP code of the stock to be studied. ticker/CUSIP must be found in data_ret columns
        estimation_period (int): number of days used to estimate the beta against the given benchmark
        before_event (int): number of days before the event to evaluate from
        event_window_start (int): a negative number specifying the relative number of days before the event date
        event_window_end (int): a positive number specifying the relative number of days after the event date
        benchmark (str): ticker symbol of the benchmark used. benchmark must be in data_ret.columns

    Returns:
        Tuple of the cumulative abnormal returns over different observation days as below
        "CAR[-1, +1]", "CAR[0,+1]", "Event Day -1", "Event Day 0", "Event Day 1",
        CAR[2,5]", "CAR[2,10]", "CAR[2,20]", "CAR[-5,-2]", "CAR[-10,-2]", "CAR[-20,-2]""
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

    # Event CAR
    winar1 = event[(event["rel_day"] <= 1) & (event["rel_day"] >= -1)][
        "abnormal_return"
    ].sum()  # CAR[-1,+1]
    winar2 = event[(event["rel_day"] <= 1) & (event["rel_day"] >= 0)][
        "abnormal_return"
    ].sum()  # CAR[0,+1]

    # Post Event CAR
    winar3 = event[(event["rel_day"] <= 5) & (event["rel_day"] >= 2)][
        "abnormal_return"
    ].sum()  # CAR[2,5]
    winar4 = event[(event["rel_day"] <= 20) & (event["rel_day"] >= 2)][
        "abnormal_return"
    ].sum()  # CAR[2,20]

    # Pre Event CAR
    winar5 = event[(event["rel_day"] <= -2) & (event["rel_day"] >= -5)][
        "abnormal_return"
    ].sum()  # CAR[-5,-2]
    winar6 = event[(event["rel_day"] <= -2) & (event["rel_day"] >= -10)][
        "abnormal_return"
    ].sum()  # CAR[-10,-2]

    return (
        winar1,
        winar2,
        winar3,
        winar4,
        winar5,
        winar6,
    )