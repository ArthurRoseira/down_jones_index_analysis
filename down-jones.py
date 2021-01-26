import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from pandas.plotting import autocorrelation_plot
import math


df = pd.read_csv('DJI.csv')
print(df.head(10))
print(df.shape)
date_time = pd.to_datetime(df['Date'])
df['Date'] = date_time
df.set_index('Date', inplace=True)
print(df.head(10))
w_moving_Avg = df['Close'].rolling(7).mean()
m_moving_Avg = df['Close'].rolling(30).mean()
print(w_moving_Avg)
w_moving_Avg.plot()
m_moving_Avg.plot()
df['Close'].plot()
plt.title('Noise Rduction with Aggregation')
plt.legend(['weekly', 'monthly', 'Original'])
plt.show()

###### First Differentiation #######
f_diff = df['Close'].diff().dropna()
print(f_diff.shape)
fig, ax = plt.subplots(2, sharex=True)
fig.set_size_inches(5.5, 5.5)
df['Close'].plot(ax=ax[0], color='b')
ax[0].set_title('Close values DJI')
f_diff.plot(ax=ax[1], color='r')
ax[1].set_title('First-order differences of DJI')
plt.show()

###### AutoCorrelation ########
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0))
ax4 = plt.subplot2grid((2, 2), (1, 1))

plot_acf(df['Close'], ax=ax1,
         title='ACF Plot StatsModels Original', unbiased=True)
ax2.set_title('ACF Plot Plt Original')
autocorrelation_plot(df['Close'], ax=ax2)
plot_acf(f_diff, ax=ax3, lags=10, unbiased=True,
         title='ACF Plot StatsModels Diff 1')
ax4.set_title('ACF Plot Plt Diff 1')
autocorrelation_plot(f_diff, ax=ax4)
plt.show()
