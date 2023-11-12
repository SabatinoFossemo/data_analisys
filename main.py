import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)


def get_data():
    d: pd.DataFrame = yf.download("SPY", progress=False)
    return d


def set_features(_in):
    _in['PctChange'] = _in['Close'].pct_change() * 100
    _in['Return'] = _in['Close'] - _in['Close'].shift(1)
    return _in


def normal_dist(series):
    d_min = series.min()
    d_max = series.max()
    d_len = series.count()
    nrm = MinMaxScaler((d_min, d_max)).fit_transform(np.random.normal(size=d_len).reshape(-1, 1))
    nrm = pd.Series(nrm[:, 0]).sort_values().reset_index(drop=True)
    return nrm


def return_plot():
    fig = plt.figure('Daily Change')
    ax1 = fig.add_subplot(211)
    ax1.title.set_text('ABS Return')
    ax1.title.set_size(8)
    plt.plot(data['Return'])
    ax2 = fig.add_subplot(212)
    ax2.title.set_text('LOG Return')
    ax2.title.set_size(8)
    plt.plot(data['PctChange'])


def history_plot():
    fig = plt.figure('SPY Historical Data')
    ax1 = fig.add_subplot(211)
    ax1.title.set_text('Close To Close Return')
    ax1.title.set_size(8)
    ax1.plot(data['Close'])

    ax2 = fig.add_subplot(212)
    ax2.title.set_text('LOG Return')
    ax2.title.set_size(8)
    ax2.plot(data['PctChange'].cumsum())


def dist_plot():
    fig = plt.figure('Distribution')

    ax1 = fig.add_subplot(211)
    ax1.title.set_text('Absolute Return vs Normal Distribution')
    ax1.title.set_size(8)
    d1 = data['Return'].sort_values().reset_index(drop=True)
    n1 = normal_dist(d1)
    ax1.hist(d1, histtype='step', bins=100)
    ax1.hist(n1, histtype='step', bins=100)
    plt.legend(['ABS Return', 'RandomNormDist'])

    ax2 = fig.add_subplot(212)
    ax2.title.set_text('Pct Return vs Normal Distribution')
    ax2.title.set_size(8)
    d1 = data['PctChange'].sort_values().reset_index(drop=True)
    n1 = normal_dist(d1)
    ax2.hist(d1, histtype='step', bins=100)
    ax2.hist(n1, histtype='step', bins=100)
    plt.legend(['PCT Return', 'RandomNormDist'])


if __name__ == '__main__':
    sns.set_theme()

    raw_data = get_data()
    data = set_features(raw_data)

    return_plot()
    history_plot()
    dist_plot()

    plt.show()
