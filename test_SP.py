import numpy as np
import pandas as pd
from model.esn import EchoStateNetwork
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import util
from dePrado import minLossFracDiff
from model.output_models import TimeSeriesCV
from sklearn.linear_model import ElasticNetCV

from statsmodels.tsa.arima_model import ARMA



df_data = pd.read_csv('data/sp500.csv')
values = df_data['Close'].rolling(200).mean().dropna().values
values = values.reshape((values.shape[0], 1))
x = np.arange(values.shape[0]).reshape((values.shape[0], 1))

de_trend = LinearRegression().fit(x, values).predict(x) - values

df_min_loss, d = minLossFracDiff(pd.DataFrame(de_trend, columns=['Price']))
#df_min_loss.plot()
#plt.show()


series = df_min_loss['Price'].values
output_model = ElasticNetCV(fit_intercept=True, cv=util.ts_split(series.shape[0]), n_alphas=10)
e = EchoStateNetwork(layered=True, initial_layer=False, output_model=output_model)
e.add_layer(layers=3, size=400, spectral_radius=0.15)
e.add_layer(size=400, bias=0.38, leak=0.87, spectral_radius=1.3)

#residuals = arma_model.geterrors()

#plt.plot(x, de_trend)
#plt.legend(['true', 'fitted', 'residual'])
#plt.show()

series = series.reshape((series.shape[0], 1))/100

burn_in_ind = int(0.15*series.shape[0])
#train_ind = int(0.995*series.shape[0])

num_test = 10

train_ind = series.shape[0] - num_test

burn_in_feature = series[:burn_in_ind, :]
train_feature = series[burn_in_ind:train_ind-1, :]
train_teacher = series[burn_in_ind+1:train_ind, :]
test_teacher = series[train_ind:, :]

num_test = test_teacher.shape[0]


e.train(burn_in_feature=burn_in_feature, feature=train_feature, teacher=train_teacher)

pred = e.predict(n_predictions=num_test, inject_error=True)
test_error = util.RMSE(pred, test_teacher)


fig = plt.figure(1)
plt.plot(np.arange(-num_test, num_test), series[-2*num_test:], 'b')
plt.plot(np.arange(-num_test, 0), e.fitted[-num_test:], 'g')
plt.plot(np.arange(-1, num_test), np.append(train_teacher[-1], pred), 'r')
plt.legend(['target', 'fitted', 'forecast'])
plt.title('Train Error = {:0.4f}, Test Error = {:0.4f}'.format(e.error, test_error))
plt.show()
