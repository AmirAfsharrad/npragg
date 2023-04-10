import numpy as np
from sklearn import svm


def svm_data_gen(K, M, n_train, dim, true_vec):
    X = np.random.rand(M, dim) * 2 - 1
    Z = np.zeros((M, 1))

    Y = np.zeros((K, M))
    for m in range(M):
        Z[m] = 1 if X[m] @ true_vec > 0 else 0

    for k in range(K):
        while True:
            X_train = np.random.rand(n_train[k], dim) * 2 - 1
            Z_train = np.zeros((n_train[k], 1))
            for n in range(n_train[k]):
                Z_train[n] = 1 if X_train[n] @ true_vec > 0 else 0
            if not (sum(Z_train) == n_train[k] or sum(Z_train) == 0):
                break

        clf = svm.SVC()
        clf.fit(X_train, Z_train.squeeze())

        Y[k, :] = clf.predict(X)

    return X, Y, Z


# X, Y, Z = svm_data_gen(K=2, M=10, n_train=[5, 5], dim=2, true_vec=np.matrix('1; 1'))
# print(Z.T)
# print(Y)
