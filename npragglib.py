import itertools
import sys

import cvxpy as cvx
import numpy as np
from scipy.optimize import minimize
from svm_data_gen import svm_data_gen


def solve_eta(X, Y, Z, K, M):
    # X: feature vector of length M
    # Y: vector matrix of size K * M
    # Z: true label vector of length M

    eta = cvx.Variable((K, M))
    temp = cvx.Variable((K, M * 2))
    # obj = get_eta_obj(X=X, Y=Y, Z=Z, K=K, M=M, eta=eta)

    obj = 0
    cons = []
    for k in range(K):
        for m in range(M):
            # print("K=", k, ", M=", m)
            obj += cvx.log_sum_exp(temp[k, m * 2: m * 2 + 2])
            cons.append(temp[k, m * 2] == 0)
            cons.append(temp[k, m * 2 + 1] ==
                        get_eta_coeffs(X=X, Y=Y, Z=Z, K=K, M=M, k_index=k, m_index=m, eta=eta, K_func=kernel))
            cons.append(eta[k, m] <= 0)

    prob = cvx.Problem(cvx.Minimize(obj), cons)
    print("Solving a convex problem for eta ...")
    prob.solve()
    print("Convex problem solution complete")

    return prob.value, eta.value


def kernel(x):
    h = 5
    return np.exp(- np.square(x) / h)


def get_eta_obj(X, Y, Z, K, M, eta):
    obj = 0
    for k in range(K):
        for m in range(M):
            obj += cvx.log_sum_exp(
                get_eta_coeffs(X=X, Y=Y, Z=Z, K=K, M=M, k_index=k, m_index=m, eta=eta, K_func=kernel))

    return obj


def get_eta_coeffs(X, Y, Z, K, M, k_index, m_index, eta, K_func):
    x = X[m_index]
    a = np.zeros((M, 1))
    for m in range(M):
        a[m] = K_func(np.linalg.norm(x - X[m]))
    a /= np.sum(a)

    if np.sign(Z[m_index]) == np.sign(Y[k_index, m_index]):
        sign = 1
    else:
        sign = -1

    out = sign * a.T @ eta[k_index, :]

    return out


def get_best_Z(X, Y, K, M):
    lst = list(itertools.product([0, 1], repeat=M))
    opt_val = 1000
    for Z in lst:
        # print(Z)
        val, eta = solve_eta(X, Y, Z, K, M)
        if val < opt_val:
            opt_val = val
            opt_eta = eta
            opt_Z = Z
    out = np.reshape(np.array(opt_Z), (M, 1))
    return out, opt_eta


def gen_BSC_dataset(K, M, eps, Z=None):
    gen_Z = False
    if Z is None:
        gen_Z = True
        Z = np.random.randint(0, 2, (M,))
    Y = np.zeros((K, M))
    for k in range(K):
        for m in range(M):
            rnd = bernoulli(eps[k])
            Y[k, m] = Z[m] if rnd == 1 else 1 - Z[m]
    if gen_Z:
        return Z, Y
    else:
        return Y


def bernoulli(p):
    if np.random.rand(1) > p:
        return 1
    else:
        return 0


def vec_norm(x, L):
    if L > 1:
        x = x.squeeze()
    out = 0
    for i in range(L):
        out += abs(x[i])
    return out


def maj_vote(Y, M, K):
    out = np.zeros((M, 1))
    for m in range(M):
        if sum(Y[:, m]) > K / 2:
            out[m] = 1
        else:
            out[m] = 0
    if M > 1:
        out = out
    return out


def f(X, M, eta, k, K_func, x):
    D = 0
    N = 0

    for m in range(M):
        N += eta[k, m] * K_func(np.linalg.norm(x - X[m]))
        D += K_func(np.linalg.norm(x - X[m]))

    return N / D


def lrt_estimator(y, lrt_coeffs, K):
    s = 0
    for k in range(K):
        s += (2 * y[k] - 1) * lrt_coeffs[k]
    if s >= 0:
        return 0
    else:
        return 1


def eps_init_point(Y, M, K):
    Y_hat = 2 * Y - 1
    R = Y_hat @ Y_hat.T / M
    t = cvx.Variable(K)
    obj = 0
    for i in range(K):
        for j in range(i + 1, K):
            obj += cvx.square(t[i] + t[j] - np.log(np.maximum(1e-9, R[i, j])))
    prob = cvx.Problem(cvx.Minimize(obj), [])
    prob.solve()
    t = t.value
    for i in range(K):
        R[i, i] = np.exp(2 * t[i])
    w, v = np.linalg.eig(R)
    idx = np.argmax(w)
    v_max = v[:, idx]
    eps = (1 - v_max) / 2
    eps /= 2 * np.max(eps)
    eps = eps[:, None]
    lrt_coeffs = np.log(eps / (1 - eps))
    Z = np.zeros((M, 1))
    for m in range(M):
        Z[m] = lrt_estimator(y=Y[:, m], lrt_coeffs=lrt_coeffs, K=K)
    return Z


def em(X, Y, M, K, p_flip=0, initialization="maj"):
    """
    :param X:
    :param Y:
    :param M:
    :param K:
    :param p_flip: each entry of the intitialization point Z=maj_vote(Y) is flipped with  probability p
    :param initialization:
    :return:
    """
    counter = 0
    iterate = True
    if initialization == "eps":
        Z = eps_init_point(Y, M, K)
    if initialization == "maj":
        Z = maj_vote(Y, M, K)
    for m in range(M):
        if np.random.rand() < p_flip:
            Z[m] = 1 - Z[m]
    while iterate:
        iterate = False
        counter += 1
        p, eta = solve_eta(X=X, Y=Y, Z=Z, M=M, K=K)
        print('likelihood value =', p)
        lrt_coeffs = np.zeros((K, M))
        for m in range(M):
            for k in range(K):
                lrt_coeffs[k, m] = f(X, M, eta, k, kernel, X[m])

        for m in range(M):
            z = lrt_estimator(y=Y[:, m], lrt_coeffs=lrt_coeffs[:, m], K=K)
            if not Z[m] == z:
                iterate = True
            Z[m] = z
    print("total EM iterations:", counter)

    return Z


def relaxed_z(X, Y, M, K):
    def get_loss(z):
        p, eta = solve_eta(X=X, Y=Y, Z=z, M=M, K=K)
        print("Eta", eta)
        t = 1
        for m in range(M):
            for k in range(K):
                epsilon = 1 / (1 + np.exp(-eta[k, m]))
                if z[m] == Y[k, m]:
                    t *= (1 - epsilon)
                else:
                    t *= epsilon
        return t

    G = []
    for m in range(M):
        G.append(lambda z, m=m: z[m])
        G.append(lambda z, m=m: 1 - z[m])
    cons = []
    for g in G:
        cons.append({'type': 'ineq', 'fun': g})
    z0 = np.ones((M, 1)) / 2
    sol = minimize(get_loss, z0, method='SLSQP', constraints=cons, options={'disp': False})
    zOpt = sol.x
    relaxed_out = ((np.sign(zOpt - .5) + 1) / 2)

    out = np.zeros((M, 1))
    for m in range(M):
        out[m] = relaxed_out[m]

    print("ZOPT", zOpt)
    return out


def train_for_theta_hat(sample_size, L, theta, ind):
    theta_hat_low = -1
    theta_hat_high = L + 1
    if L == 1:
        theta_hat_high = 1
    neg_count = 0

    for m in range(sample_size):
        if L == 1:
            X = np.random.rand() * 2 - 1
        else:
            X = np.random.randint(0, L - 1)

        Z = 1 if X > theta else 0

        if Z == 0 and X > theta_hat_low:
            theta_hat_low = X
            neg_count += 1

        if Z == 1 and X < theta_hat_high:
            theta_hat_high = X

    if ind == 0:
        return theta_hat_low
    if ind == 1:
        return theta_hat_high
    if ind == 2:
        if L == 1:
            out = np.random.rand() * (theta_hat_high - theta_hat_low) + theta_hat_low
        else:
            out = np.random.randint(theta_hat_low, theta_hat_high)

        print((theta_hat_low, theta_hat_high), out)
        return out
    if ind == 3:
        if sample_size == 0:
            theta_hat_2 = 0
        else:
            theta_hat_2 = neg_count / sample_size
        if theta_hat_2 < theta_hat_low:
            return theta_hat_low
        if theta_hat_2 > theta_hat_high:
            return theta_hat_high
        return theta_hat_2


def gen_x_based_dataset(M, K, dim, X=None, sim=0):
    gen_X = False
    if X is None:
        gen_X = True
        X = np.random.rand(M, dim) * 2 - 1
    Z = np.zeros((M, 1))
    Y = np.zeros((K, M))

    if sim == 0:
        for m in range(M):
            Z[m] = 1 if sum(X[m]) > 0 else 0
            for k in range(K):
                if k == 0:
                    rnd = bernoulli(.1)
                if k == 1:
                    if sum(X[m]) > 0:
                        rnd = bernoulli(.1)
                    else:
                        rnd = bernoulli(.45)
                if k == 2:
                    if sum(X[m]) > 0:
                        rnd = bernoulli(.45)
                    else:
                        rnd = bernoulli(.1)
                if k > 2:
                    rnd = bernoulli(.25)
                Y[k, m] = Z[m] if rnd == 1 else 1 - Z[m]

    if sim == 1:
        for m in range(M):
            eps = [.2, .25, .3, .3, .4]
            Z[m] = 1 if sum(X[m]) > 0 else 0
            for k in range(K):
                if 0.5 > sum(X[m]) > -.5:
                    Y[k, m] = Z[m]
                else:
                    rnd = bernoulli(eps[k])
                    Y[k, m] = Z[m] if rnd == 1 else 1 - Z[m]

    if sim == 2:
        # X = np.random.rand(M, dim) * 2 - 1
        D = 100
        eps_plus = [.05, .05, .45, .45, .25]
        # eps_minus = [.45, .45, .05, .05, .25]
        # eps_plus = [.25, .25, .25, .25, .25]
        eps_minus = eps_plus
        for m in range(M):
            Z[m] = 1 if sum(X[m]) > 0 else 0
            if m % 2 == 0:
                X[m] += D
            else:
                X[m] -= D

            for k in range(K):
                if sum(X[m]) > 0:
                    rnd = bernoulli(eps_plus[k])
                else:
                    rnd = bernoulli(eps_minus[k])

                Y[k, m] = Z[m] if rnd == 1 else 1 - Z[m]

    if sim == 7:
        # X = np.random.rand(M, dim) * 2 - 1
        eps_plus = [.05, .05, .45, .45, .45]
        eps_minus = [.01, .01, .01, .01, .01]
        # eps_plus = [.25, .25, .25, .25, .25]
        # eps_minus = eps_plus
        counter = 0
        for m in range(M):
            Z[m] = 1 if sum(X[m]) > 0 else 0

            for k in range(K):
                if sum(X[m]) < 1 and sum(X[m]) > -1:
                    counter += 1
                    rnd = bernoulli(eps_plus[k])
                else:
                    rnd = bernoulli(eps_minus[k])

                Y[k, m] = Z[m] if rnd == 1 else 1 - Z[m]

    if sim == 3:
        M_train = [2, 2, 2, 20, 20]
        theta = np.random.rand() * 2 - 1
        # theta = 0

        X = np.random.rand(M) * 2 - 1
        Z = np.zeros((M, 1))

        for m in range(M):
            Z[m] = 1 if X[m] > theta else 0

        theta_hat = np.zeros((K, 1))
        Y = np.zeros((K, M))

        for k in range(K):
            theta_hat[k] = train_for_theta_hat(sample_size=M_train[k], L=1, theta=theta, ind=2)
            for m in range(M):
                Y[k, m] = 1 if X[m] > theta_hat[k] else 0
                Y[k, m] = Y[k, m] if np.random.rand() > M_train[k] / 40 else 1 - Z[m]

    if sim == 4:
        L = 100
        M_train = [2, 2, 2, 20, 20]
        theta = np.random.randint(0, L - 1)

        X = np.random.randint(0, L - 1, M)
        Z = np.zeros((M, 1))
        for m in range(M):
            Z[m] = 1 if X[m] > theta else 0

        theta_hat = np.zeros((K, 1))
        Y = np.zeros((K, M))

        for k in range(K):
            theta_hat[k] = train_for_theta_hat(M_train[k], L, theta, 2)
            for m in range(M):
                Y[k, m] = 1 if X[m] > theta_hat[k] else 0

    if sim == 5:
        D = 0
        for m in range(M):
            Z[m] = 1 if sum(X[m]) > 0 else 0
            if m % 2 == 0:
                X[m] += D
            else:
                X[m] -= D

        noise_level = [5, 5, 5, 5, 5]

        for k in range(K):
            a = np.ones((1, dim)) + (np.random.randn(1, dim)) * noise_level[k]
            for m in range(M):
                if X[m] @ a.T > 0:
                    Y[k, m] = 1
                else:
                    Y[k, m] = 0

    if sim == 6:
        X, Y, Z = svm_data_gen(K=K, M=M, n_train=[3, 3, 3, 100, 100], dim=dim, true_vec=np.ones((dim, 1)))
        print(Z.T)
        print(Y)

    if gen_X:
        return X, Y, Z
    else:
        return Y, Z
