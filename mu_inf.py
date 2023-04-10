import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True


def unif(a, b):
    return (b - a) * np.random.rand() + a


def threshold(X, p):
    return 1 if X > p else -1


def getXandZ(theta, epsilon):
    X = unif(-1, 1)
    Z = threshold(X, theta)
    if np.random.rand() < epsilon:
        Z = -Z
    return X, Z


def train_for_theta_hat(sample_size, theta):
    theta_hat = -1
    for m in range(sample_size):
        X, Z = getXandZ(theta, 0)
        if Z == -1 and X > theta_hat:
            theta_hat = X

    return theta_hat


def getXindex(X):
    return int(nX_bins * (X + 1) / 2)


epsilon = .1
nX_bins = 100
# P(Y1,Y2|X,Z)
P12 = np.zeros((2, 2, nX_bins, 2))
# P(Y1|X,Z)
P1 = np.zeros((2, nX_bins, 2))
# P(Y2|X,Z)
P2 = np.zeros((2, nX_bins, 2))
Pxz = np.zeros((nX_bins, 2))

N = 1000000
M_range = 5
MI = np.zeros((M_range, 1))
for M in range(M_range):
    # os.system('clear')
    print("M = ", M)
    for i in range(N):
        # if i % 10000 == 0:
        #     os.system('clear')
        #     print(str(i / 10000 + 1) + " out of " + str(N / 10000))
        theta = unif(-1, 1)
        # theta = 0
        theta_hat1 = train_for_theta_hat(M, theta)
        theta_hat2 = train_for_theta_hat(M, theta)

        X, Z = getXandZ(theta, epsilon)
        Y1 = threshold(X, theta_hat1)
        Y2 = threshold(X, theta_hat2)

        y1_index = int((Y1 + 1) / 2)
        y2_index = int((Y2 + 1) / 2)
        x_index = getXindex(X)
        z_index = int((Z + 1) / 2)
        P12[y1_index, y2_index, x_index, z_index] += 1
        P1[y1_index, x_index, z_index] += 1
        P2[y2_index, x_index, z_index] += 1
        Pxz[x_index, z_index] += 1

    for x_index in range(nX_bins):
        for z_index in range(2):
            if Pxz[x_index, z_index] > 0:
                for y1_index in range(2):
                    P1[y1_index, x_index, z_index] /= Pxz[x_index, z_index]
                    P2[y1_index, x_index, z_index] /= Pxz[x_index, z_index]
                    for y2_index in range(2):
                        P12[y1_index, y2_index, x_index, z_index] /= Pxz[x_index, z_index]

    s = 0
    delta = 2 / nX_bins
    for x_index in range(nX_bins):
        for z_index in range(2):
            for y1_index in range(2):
                for y2_index in range(2):
                    p1p2 = P1[y1_index, x_index, z_index] * P2[y2_index, x_index, z_index]
                    p12 = P12[y1_index, y2_index, x_index, z_index]
                    if p1p2 > 0 and p12 > 0:
                        s += p12 * np.log2(p12 / p1p2) * delta

    MI[M] = s
    print(s)

# print("mutual information", s)

    plt.figure()
    plt.plot(np.linspace(1, M_range, M_range), MI, 'b', label='eps = ' + str(epsilon))
    plt.xlabel("$M$")
    plt.ylabel("$I(Y_1,Y_2)$")
    plt.title("Mutual Information of $Y_1, Y_2$ for $\epsilon = " + str(epsilon) + "$ vs. Training Sample Size $M$")
    plt.show()
