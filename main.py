import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)


sns.set_theme()

def get_data(symbol) -> pd.DataFrame:
    """'
    download SPY Data from Yahoo Finance
    :return: SPY History as pandas.Dataframe
    """
    d: pd.DataFrame = yf.download(symbol, progress=False)
    return d


def set_features(_in) -> pd.DataFrame:
    """
    Add columns for Percent Log return and Absolute Return
    :param _in: raw_data
    :return: Dataframe with additional Features
    """
    _in['PctChange'] = _in['Close'].pct_change() * 100
    _in['Return'] = _in['Close'] - _in['Close'].shift(1)

    return _in


def normal_dist(series: pd.Series) -> pd.Series:
    d_min = series.min()
    d_max = series.max()
    d_len = series.count()
    nrm = MinMaxScaler((d_min, d_max)).fit_transform(np.random.normal(size=d_len).reshape(-1, 1))
    nrm = pd.Series(nrm[:, 0]).sort_values().reset_index(drop=True)
    return nrm


def primary_swing(data):

    swing_high = (data['High'] > data['High'].shift(1)) & (data['High'] > data['High'].shift(-1))
    swing_low = (data['Low'] < data['Low'].shift(1)) & (data['Low'] < data['Low'].shift(-1))
    high = data['High']
    low = data['Low']
    date = data.index

    trend = 0
    result = []
    index = []
    sw_len = []
    sw_type = []
    for sh, sl, h, l, d in zip(swing_high, swing_low, high, low, date):
        if trend < 0:
            if sh:
                result.append(h)
                index.append(d)
                sw_len.append(-trend)
                sw_type.append('H')
                trend = 1
            else:
                trend -= 1

        elif trend > 0:
            if sl:
                result.append(l)
                index.append(d)
                sw_len.append(trend)
                sw_type.append('L')
                trend = -1
            else:
                trend += 1
        else:
            if sh:
                result.append(h)
                index.append(d)
                sw_len.append(-trend)
                sw_type.append('H')
                trend = 1
            elif sl:
                result.append(l)
                index.append(d)
                sw_len.append(trend)
                sw_type.append('L')
                trend = -1
            else:
                continue

    col = ['Swing', 'Length', 'Type']
    result = pd.DataFrame(zip(result, sw_len, sw_type), index=index, columns=col)
    result['AbsRange'] = result['Swing'].diff()
    result['PctRange'] = result['Swing'].pct_change() * 100
    return result


def plot_primary_swing(swing_chart, linewidth):
    fig = plt.figure('Primary Swing Chart')
    fig.tight_layout()

    ax1 = plt.subplot(221)
    ax1.title.set_text('ABS Chart')
    ax1.plot(swing_chart['Swing'], linewidth=linewidth)

    ax2 = plt.subplot(222)
    ax2.title.set_text('PCT Chart')
    ax2.plot(swing_chart['PctRange'].cumsum(), linewidth=linewidth)

    ax3 = plt.subplot(223)
    ax3.title.set_text('ABS Range')
    ax3.plot(swing_chart['AbsRange'], linewidth=linewidth)

    ax4 = plt.subplot(224)
    ax4.title.set_text('PCT  Range')
    ax4.plot(swing_chart['PctRange'], linewidth=linewidth)

    return fig


def secondary_swing(swing):
    pass


def history_plot(data, linewidth):
    """
    plot Historical data both Absolute and Log
    :return:
    """
    fig = plt.figure('Historical Data')

    ax1 = fig.add_subplot(221)
    ax1.title.set_text('Close To Close Return')
    ax1.plot(data['Close'], linewidth=linewidth)

    ax2 = fig.add_subplot(222)
    ax2.title.set_text('LOG Chart')
    ax2.plot(data['PctChange'].cumsum(), linewidth=linewidth)

    ax3 = fig.add_subplot(223)
    ax3.title.set_text('ABS Return')
    plt.plot(data['Return'], linewidth=linewidth)

    ax4 = fig.add_subplot(224)
    ax4.title.set_text('LOG Return')
    plt.plot(data['PctChange'], linewidth=linewidth)

    return fig


def dist_plot(data):
    """
    plot Distribution of returns vs normal distribution
    :return:
    """
    fig = plt.figure('Distribution')

    ax1 = fig.add_subplot(211)
    ax1.title.set_text('Absolute Return vs Normal Distribution')
    d1 = data['Return'].sort_values().reset_index(drop=True)
    n1 = normal_dist(d1)
    ax1.hist(d1, histtype='step', bins=100)
    ax1.hist(n1, histtype='step', bins=100)
    plt.legend(['ABS Return', 'RandomNormDist'])

    ax2 = fig.add_subplot(212)
    ax2.title.set_text('Pct Return vs Normal Distribution')
    d1 = data['PctChange'].sort_values().reset_index(drop=True)
    n1 = normal_dist(d1)
    ax2.hist(d1, histtype='step', bins=100)
    ax2.hist(n1, histtype='step', bins=100)
    plt.legend(['PCT Return', 'RandomNormDist'])

    return fig


def plt_setting(figures):
    for fig in figures:
        for axis in fig.axes:
            axis.tick_params(labelsize=6)
            axis.title.set_size(8)


def main():
    symbol = "SPY"
    raw_data = get_data(symbol)
    data = set_features(raw_data)
    swing_chart = primary_swing(data).reset_index(drop=True)

    print(swing_chart)
    print(swing_chart.describe())
    plots = []
    fig = plot_primary_swing(swing_chart, linewidth='.6')
    plots.append(fig)
    fig = history_plot(data, linewidth='.6')
    plots.append(fig)

    fig = dist_plot(data)
    plots.append(fig)

    plt_setting(plots)

    plt.show()

if __name__ == '__main__':
    main()
