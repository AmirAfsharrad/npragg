import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True


def threshold(X, p):
    return 1 if X > p else 0


def getXandZ(theta, L, epsilon):
    X = np.random.randint(0, L - 1)
    Z = threshold(X, theta)
    if np.random.rand() < epsilon:
        Z = 1 - Z
    return X, Z


def train_for_theta_hat(sample_size, L, theta, ind):
    theta_hat_low = 0
    theta_hat_high = L + 1
    neg_count = 0

    for m in range(sample_size):
        X, Z = getXandZ(theta, L, 0)

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
        out = np.random.randint(theta_hat_low, theta_hat_high)
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

    # out = np.random.rand() * (theta_hat_high - theta_hat_low) + theta_hat_low
    # # return out
    # return (theta_hat_low + theta_hat_high) / 2


# epsilon = .5
L = 100
N = 10000
# N = 10000
M_range = 30
M_step = 5
eps_N = 1
MIxz = np.zeros((M_range, eps_N))
MIx = np.zeros((M_range, eps_N))
eps_vector = np.linspace(0, 0, eps_N)
for epsilon_index in range(eps_N):
    epsilon = eps_vector[epsilon_index]
    print(epsilon)
    for M in range(M_range):
        M = M * M_step
        # P(Y1,Y2,X,Z)
        P12xz = np.zeros((2, 2, L, 2))
        # P(Y1,Y2|X,Z)
        P12_xz = np.zeros((2, 2, L, 2))
        # P(Y1,X,Z)
        P1xz = np.zeros((2, L, 2))
        # P(Y1|X,Z)
        P1_xz = np.zeros((2, L, 2))
        # P(Y2,X,Z)
        P2xz = np.zeros((2, L, 2))
        # P(Y2|X,Z)
        P2_xz = np.zeros((2, L, 2))

        # P(Y1,Y2,X)
        P12x = np.zeros((2, 2, L))
        # P(Y1,Y2|X)
        P12_x = np.zeros((2, 2, L))
        # P(Y1,X)
        P1x = np.zeros((2, L))
        # P(Y1|X)
        P1_x = np.zeros((2, L))
        # P(Y2,X)
        P2x = np.zeros((2, L))
        # P(Y2|X)
        P2_x = np.zeros((2, L))

        Pxz = np.zeros((L, 2))
        Px = np.zeros((L, 1))


        print("M = ", M)
        for i in range(N):
            if i % 1000 == 0:
                print("N", i / 1000)
            theta = np.random.randint(0, L - 1)
            # theta = 50
            theta_hat1 = train_for_theta_hat(M, L, theta, 0)
            theta_hat2 = train_for_theta_hat(M, L, theta, 0)

            x, z = getXandZ(theta, L, epsilon)
            y1 = threshold(x, theta_hat1)
            y2 = threshold(x, theta_hat2)

            P12xz[y1, y2, x, z] += 1 / N
            P1xz[y1, x, z] += 1 / N
            P2xz[y2, x, z] += 1 / N
            Pxz[x, z] += 1 / N

            P12x[y1, y2, x] += 1 / N
            Px[x] += 1 / N
            P1x[y1, x] += 1 / N
            P2x[y2, x] += 1 / N

        for x in range(L):
            for z in range(2):
                if Pxz[x, z] > 0:
                    for y1 in range(2):
                        P1_xz[y1, x, z] = P1xz[y1, x, z] / Pxz[x, z]
                        P2_xz[y1, x, z] = P2xz[y1, x, z] / Pxz[x, z]

                        for y2 in range(2):
                            P12_xz[y1, y2, x, z] = P12xz[y1, y2, x, z] / Pxz[x, z]

        for x in range(L):
            if Px[x] > 0:
                for y1 in range(2):
                    P1_x[y1, x] = P1x[y1, x] / Px[x]
                    P2_x[y1, x] = P2x[y1, x] / Px[x]
                    for y2 in range(2):
                        P12_x[y1, y2, x] = P12x[y1, y2, x] / Px[x]


        s = np.zeros((L, 2))
        sx = np.zeros((L, 1))
        ss = 0

        for x in range(L):
            for z in range(2):
                for y1 in range(2):
                    for y2 in range(2):
                        p1p2 = P1_xz[y1, x, z] * P2_xz[y2, x, z]
                        p12 = P12_xz[y1, y2, x, z]
                        if p1p2 > 0 and p12 > 0:
                            s[x, z] += p12 * np.log2(p12 / p1p2)
                            ss += P12xz[y1, y2, x, z] * np.log2(p12 / p1p2)
                            print(P12xz[y1, y2, x, z], "oooo")

        ss2 = 0
        for x in range(L):
            for y1 in range(2):
                for y2 in range(2):
                    p1p2 = P1_x[y1, x] * P2_x[y2, x]
                    p12 = P12_x[y1, y2, x]
                    if p1p2 > 0 and p12 > 0:
                        # sx[x] += p12 * np.log2(p12 / p1p2)
                        ss2 += P12x[y1, y2, x] * np.log2(p12 / p1p2)

        MIx[int(M / M_step), epsilon_index] = ss2
        MIxz[int(M / M_step), epsilon_index] = ss
        # MIxz[int(M / M_step), epsilon_index] = np.mean(s)
        # print(np.mean(sx))

    # print("mutual information", s)

plt.figure()
for epsilon_index in range(eps_N):
    # plt.plot(M_step * np.linspace(1, M_range, M_range), MI[:, epsilon_index],
    #          label='eps = ' + str(eps_vector[epsilon_index]))
    # plt.plot(M_step * np.linspace(0, M_range - 1, M_range), MIx[:, epsilon_index],
    #          label=r"$I(Y_1,Y_2|X), \epsilon = " + str(eps_vector[epsilon_index]) + "$")
    plt.plot(M_step * np.linspace(0, M_range - 1, M_range), MIxz[:, epsilon_index],
             label=r"$I(Y_1,Y_2|X,Z), \epsilon = " + str(eps_vector[epsilon_index]) + "$")
plt.grid(which='both')
plt.legend()
plt.xlabel("$M$")
plt.ylabel("$I(Y_1,Y_2|X,Z)$")
plt.title("Mutual Information of $Y_1, Y_2$ vs. Training Sample Size $M$")
plt.show()
