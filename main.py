import numpy as np
import npragglib as lib
import sim
import npragglib as npr
import numpy as np

np.random.seed(1)

# sim.em_sim(K=5, M=20, N=100)

# X, Y, Z = npr.gen_x_based_dataset(M=5, K=5, dim=2)

# print(X)
# print(Z)
# print(Y)
X = np.matrix(" 10; 0; 0; 0")
Y = np.matrix("1, 0, 0, 0;"
              "0, 0, 0, 0;"
              "0, 0, 0, 0;"
              "1, 0, 0, 0;"
              "1, 1, 1, 1")
# Z, eta = lib.get_best_Z(X, Y, K=5, M=4)
Zem = lib.em(X, Y, K=5, M=4, p_flip=0)
# print(eta)
# print(Z)
print(Zem)
# sim.optimal_sim(K=5, M=10, N=20)
# sim.em_sim_x(K=5, M=500, N=50, dim=10, sim=7, delta=100, run_em0=False, plot_local=False)
# sim.em_sim(K=5, M=200, N=10, delta=40)
