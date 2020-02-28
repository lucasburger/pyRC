import numpy as np
import numba


def MackeyGlass(l, beta=0.2, gamma=0.1, tau=17, n=10, drop_out=0.1, initial_condition=None, random_seed=None, random=False):
    if random_seed or initial_condition is None:
        np.random.seed(random_seed)
        initial_condition = np.random.uniform(0.5, 1.2, tau)
    elif not isinstance(initial_condition, np.ndarray):
        initial_condition = np.asarray(initial_condition).flatten()

    if initial_condition.shape[0] != tau:
        initial_condition = np.hstack((initial_condition, np.zeros((tau-initial_condition.shape[0],))))

    assert len(initial_condition) >= tau
    assert l > tau

    l = int(l/(1-drop_out))

    mg = MackeyGlassNumba(l, initial_condition, beta=beta, gamma=gamma, tau=tau, n=n)
    return mg[int(l*drop_out):]


@numba.njit(parallel=True)
def MackeyGlassNumba(l: int, initial_condition: np.ndarray, beta: float = 0.2, gamma: float = 0.1, tau: int = 17, n: int = 10):

    x = np.zeros(shape=(l,))
    x[:tau] = initial_condition[-tau:]

    for i in range(tau, l):
        x[i] = x[i-1] + beta*x[i-tau]/(1+x[i-tau]**n) - gamma*x[i-1]

    return x


# this just compiles MackeyGlassNumba to be used later on
x = MackeyGlass(100)
