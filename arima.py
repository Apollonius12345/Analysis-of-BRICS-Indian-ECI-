import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import itertools

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from mpl_toolkits import mplot3d

sns.set(style="whitegrid")
warnings.filterwarnings("ignore")

pip install pmdarima

df = pd.read_csv("/content/Country Complexity Rankings 1995 - 2020 (3).csv", index_col='Date', parse_dates=True)
# df.T.head()
# df = df.dropna()
df.T.head()
df

df.shape

from statsmodels.tsa.stattools import adfuller
def ad_test(dataset):
  dftest = adfuller(dataset, autolag='AIC')
  print("P-value :", dftest[0])
  for key,val in dftest[1].items():
    print(key, ":", val)

df['India'].plot(figsize=(12,5))

adfuller(df['India'], autolag = 'AIC')

from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")

stepwise_fit = auto_arima(df['India'], trace = True, supress_warnings=True)
stepwise_fit.summary()

from statsmodels.tsa.arima_model import ARIMA

print(df.shape)
train = df.iloc[:-10]
test = df.iloc[-10:]
print(train.shape, test.shape)

import statsmodels.api as sm
model = sm.tsa.arima.ARIMA(train['India'], order =(100,3,4))
model = model.fit()
model.summary()

start = len(train)
end = len(train)+len(test)
pred = model.predict(start=start, end=end, typ = 'levels')
print(pred)

pred.plot(legend = True)
test['India'].plot(legend = True)

df['India'].plot(figsize=(12,5))
df['South Africa'].plot()
df['Russia'].plot()
df['China'].plot()
df['Brazil'].plot()
plt.legend(loc="upper left")
