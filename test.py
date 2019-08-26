
from model.esn import EchoStateNetwork as ESN, deepESN
from util import MackeyGlass, RMSE
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
np.random.seed(42)



l = 10000 # length of mackey glass timeseries
mg = MackeyGlass(l)
mg = np.array(mg[int(0.1*l):]) # drop initial observations to wash out random initialisation
mg -= np.mean(mg) # demean
mg = mg.reshape((mg.shape[0], 1)) # reshape to (9000, 1)


####### split into burn_in, train and test parts
num_test = 1000
burn_in_ind = int(0.15*mg.shape[0])
burn_in_feature = mg[:burn_in_ind]
train_feature = mg[burn_in_ind:-num_test-1]
train_teacher = mg[burn_in_ind+1:-num_test]
test_teacher = mg[-num_test:]


# set up ESN
size = 2000
num_layers = 0
size = int(size/(1 if num_layers == 0 else num_layers))
e = ESN(size=size, bias=0.38, leak=0.87, sparsity=0.97, spectral_radius=1.3)
for i in range(num_layers):
    e.add_layer(size=size, bias=0.38, leak=0.87, sparsity=0.97, spectral_radius=1.3, input=(i==0))
    if i > 0:
        e.reservoir.set_connection(layer_from=i-1, layer_to=i, sparsity=0.97, spectral_radius=0.8)

n = dt.now()
e.train(burn_in_feature=burn_in_feature, feature=train_feature, teacher=train_teacher)
print("Training time:", dt.now()-n)

pred = e.predict(n_predictions=num_test, inject_error=False, simulation=True)
test_error = RMSE(pred, test_teacher)

fig = plt.figure(1)
plt.subplot(121)
plt.plot(np.arange(-int(0.5*num_test), num_test), mg[-int(1.5*num_test):], 'b')
plt.plot(np.arange(-int(0.5*num_test), 0), e.fitted[-int(0.5*num_test):], 'g')
plt.plot(np.arange(-1, num_test), np.append(train_teacher[-1], pred), 'r')
plt.legend(['target', 'fitted', 'forecast'])
plt.title('Train Error = {:0.6f}, Test Error = {:0.6f}'.format(e.error, test_error))

#plt.show(block=False)

ax = plt.subplot(122)
plt.imshow(e.reservoir.echo.copy(), cmap='bwr')
plt.show(block=False)
plt.xticks([])
plt.yticks([])
plt.savefig('figures/{}_mg_{}layer.png'.format(dt.now().strftime("%Y%m%d%H%M"), num_layers))

