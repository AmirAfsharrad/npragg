import numpy as np
import pickle
import npragglib as lib

K = 5
M = 500
X0 = np.zeros((M, 1))
Y = np.zeros((K, M))
for k in range(4):
    y = pickle.load(open('y_pred' + str(k + 1) + '.pkl', 'rb'))
    Y[k, :] = y

Z = pickle.load(open('hot_dog_test_label.pkl', 'rb'))
X = pickle.load(open('hot_dog_test.pkl', 'rb')).T
# print(X[0].shape)
# mx = 10000
# for i in range(M):
#     print(X[i])
    # if mx > lib.vec_norm(X[i], M):
    #     mx = lib.vec_norm(X[i], M)
    #     xmax = X[i]
# [print(xmax[j]) for j in range(M)]
maj_est = lib.maj_vote(Y=Y[:4, :], M=M, K=4)
em_z = lib.em(X=X, Y=Y[:4, :], M=M, K=4)
# em_z0 = lib.em(X=X0, Y=Y[:4, :], M=M, K=4)

print(1 - lib.vec_norm(Z - maj_est, M) / M)
print(1 - lib.vec_norm(Z - em_z, M) / M)