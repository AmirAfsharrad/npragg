import matplotlib.pyplot as plt
import numpy as np

theta = np.random.rand()
theta = .8
N = 20
N_iter = 100000
t = np.zeros((N_iter, 1))

for n_iter in range(N_iter):
    l = 0
    u = 1

    for i in range(N):
        x = np.random.rand()
        if x > theta:
            if x < u:
                u = x
        else:
            if x > l:
                l = x

    t[n_iter] = np.random.rand() * (u - l) + l

x = np.linspace(0, 1, N_iter)
s = np.sqrt(np.var(t) / 2)
y = np.exp(-abs(x - theta) / s) / (2 * s)

plt.figure()
plt.hist(t, bins=60, density=True)
plt.plot(x, y)
plt.show()
