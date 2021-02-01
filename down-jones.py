import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose
import math
from scipy.stats import boxcox


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
plt.title('Noise Reduction with Aggregation')
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
plot_acf(f_diff, ax=ax3, unbiased=True,
         title='ACF Plot StatsModels Diff 1')
ax4.set_title('ACF Plot Plt Diff 1')
autocorrelation_plot(f_diff, ax=ax4)
plt.show()

### Seazonality Verification #######
# Weekly AutoCorrelation
plot_acf(df['Close'], lags=7,
         title='ACF Plot StatsModels Original', unbiased=True)
season = seasonal_decompose(df['Close'], period=7)
fig = season.plot()
fig.set_size_inches(16, 8)
plt.show()

#### Stationarity Test ######
#### acf plot ####
plot_acf(f_diff, title='First Differentiation', unbiased=True)
plt.show()
adfinput = adfuller(f_diff, autolag='AIC')
adftest = pd.Series(adfinput[0:4], index=['Teste Estatistico Dickey Fuller',
                                          'Valor-P', 'Lags Usados', 'Número de observações usadas'])
adftest = round(adftest, 4)
for key, value in adfinput[4].items():
    adftest["Valores Críticos (%s)" % key] = value.round(4)
print(adftest)
# KPSS Test
kpss_input = kpss(f_diff, regression='c', nlags="auto")
kpss_test = pd.Series(kpss_input[0:3], index=[
                      'Teste Statistico KPSS', 'Valor-P', 'Lags Usados'])
kpss_test = round(kpss_test, 4)
for key, value in kpss_input[3].items():
    kpss_test["Valores Críticos (%s)" % key] = value
print(kpss_test)
season = seasonal_decompose(f_diff, period=30)
fig = season.plot()
fig.set_size_inches(16, 8)
plt.show()
# The p-value is obtained is greater than significance level of 0.05 and the ADF statistic is higher than any of the critical values.
# Kpps is stationary

#### Applying Box Cox  Transformation in  Data #########
# Data must be positive
boxcox_transform = boxcox(df['Close'], -0.5)
#boxcox_transform = pd.Series(boxcox_transform).diff().dropna()
plot_acf(boxcox_transform, title='First Differentiation', unbiased=True)
plt.show()
adfinput = adfuller(boxcox_transform, autolag='AIC')
adftest = pd.Series(adfinput[0:4], index=['Teste Estatistico Dickey Fuller',
                                          'Valor-P', 'Lags Usados', 'Número de observações usadas'])
adftest = round(adftest, 4)
for key, value in adfinput[4].items():
    adftest["Valores Críticos (%s)" % key] = value.round(4)
print(adftest)
# KPSS Test
kpss_input = kpss(boxcox_transform, regression='c', nlags="auto")
kpss_test = pd.Series(kpss_input[0:3], index=[
                      'Teste Statistico KPSS', 'Valor-P', 'Lags Usados'])
kpss_test = round(kpss_test, 4)
for key, value in kpss_input[3].items():
    kpss_test["Valores Críticos (%s)" % key] = value
print(kpss_test)
season = seasonal_decompose(boxcox_transform, period=30)
fig = season.plot()
fig.set_size_inches(16, 8)
plt.show()
