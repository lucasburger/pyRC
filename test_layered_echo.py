from model.esn import EchoStateNetwork
import numpy as np
import util
import matplotlib.pyplot as plt

np.random.seed(42)

size = 10
e = EchoStateNetwork(layered=True, initial_layer=False)
for _ in range(3):
    e.add_layer(size=size, leak=1, spectral_radius=-1, output_connection=False, connect_from=[], activation=util.identity)

e.add_layer(size=size, bias=0.38, leak=0.87, spectral_radius=1.3)


for i in range(3):
    e.reservoir.add_connection(layer_from=i, layer_to=i+1, matrix=np.eye(size))

#print(e.reservoir.echo)

#plt.imshow(e.reservoir.echo, cmap='hot')
#plt.colorbar()
#plt.show()


for i, m in e.reservoir.connections.items():
    print(i, m)


for i, m in enumerate(e.reservoir.layers):
    print(i, m.echo)
