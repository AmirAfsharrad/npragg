import itertools
import numpy as np
import matplotlib.pyplot as plt
import npragglib as lib
plt.rcParams['text.usetex'] = True


def optimal_sim(K, M, N):
    Z = np.random.randint(0, 2, (M, 1))
    eps = [.05, .05, .3, .4, .4]
    maj_acc = np.zeros((M, 1))
    opt_acc = np.zeros((M, 1))
    em_acc = np.zeros((M, 1))

    for n in range(1, N + 1):
        print(n)
        for m in range(1, M + 1, M - 1):
            print("m = ", m)
            Y = lib.gen_BSC_dataset(K=K, M=m, eps=eps, Z=Z)
            X = np.linspace(0, 1, M) * 0
            maj_z = lib.maj_vote(Y=Y, M=m, K=K)
            opt_z, opt_eta = lib.get_best_Z(X=X, Y=Y, K=K, M=m)
            em_z = lib.em(X=X, Y=Y, K=K, M=m)
            maj_acc[m - 1] += (1 - np.linalg.norm(Z[:m] - maj_z, 1) / m) / N
            opt_acc[m - 1] += (1 - np.linalg.norm(Z[:m] - opt_z, 1) / m) / N
            em_acc[m - 1] += (1 - np.linalg.norm(Z[:m] - em_z, 1) / m) / N
            # print("maj: ", maj_z.T, maj_acc[m - 1])
            # print("opt: ", opt_z, opt_acc[m - 1])

    fig = plt.figure()
    plt.plot(maj_acc)
    plt.plot(opt_acc)
    plt.plot(em_acc)
    plt.legend(("maj", "opt", "em"))
    plt.show()
    fig.savefig('result.eps')


def em_sim(K, M, N, delta=1):
    Z = np.random.randint(0, 2, (M, 1))
    l = int(M / delta)
    eps = [.05, .05, .3, .4, .4]
    maj_acc = np.zeros((l, 1))
    em_acc = np.zeros((l, 1))

    for n in range(1, N + 1):
        print(n)
        Y = lib.gen_BSC_dataset(K=K, M=M, eps=eps, Z=Z)
        for m in range(1, M + 1, delta):
            print(m)

            X = np.linspace(0, 1, M) * 0
            maj_z = lib.maj_vote(Y=Y, M=m, K=K)
            try:
                em_z = lib.em(X=X, Y=Y, M=m, K=K)
                index = int((m - 1) / delta)
                maj_acc[index] += (1 - np.linalg.norm(Z[:m] - maj_z, 1) / m) / N
                em_acc[index] += (1 - np.linalg.norm(Z[:m] - em_z, 1) / m) / N

            except Exception:
                pass

            # print("maj: ", maj_z.T, maj_acc[m - 1])
            # print("opt: ", opt_z, opt_acc[m - 1])

        m_vector = np.linspace(1, M, l)
        normalizer_coeff = N / n

        fig = plt.figure()
        plt.plot(m_vector, maj_acc * normalizer_coeff, 'b')
        plt.plot(m_vector, em_acc * normalizer_coeff, '--r')
        plt.legend(("maj", "EM"))
        plt.title("N = " + str(n))
        plt.show()
        fig.savefig('result.eps')


def em_sim_x(K, M, N, dim, sim, delta=1, run_em0=False, plot_local=False):
    X = np.random.rand(M, dim) * 2 - 1
    l = int(M / delta)
    maj_acc = np.zeros((l, 1))
    em_acc = np.zeros((l, 1))
    if run_em0:
        em0_acc = np.zeros((l, 1))
    loc_acc = np.zeros((l, K))
    error_occ = np.zeros((l, 1))
    n_vec = np.zeros((l, 1))

    for n in range(1, N + 1):
        print(n)
        n_vec += 1
        X, Y, Z = lib.gen_x_based_dataset(M=M, K=K, dim=dim, sim=sim)
        # Z = lib.eps_init_point(Y, M, K)
        # X = X / 100
        X0 = X * 0
        error_occ *= 0
        for m in range(1, M + 1, delta):
            print(m)
            maj_z = lib.maj_vote(Y=Y, M=m, K=K)
            index = int((m - 1) / delta)
            # em_z0 = lib.relaxed_z(X=X, Y=Y, M=m, K=K)
            try:
                em_z = lib.em(X=X, Y=Y, M=m, K=K, p_flip=0., initialization="maj")
                if run_em0:
                    em_z0 = lib.em(X=X*0, Y=Y, M=m, K=K, p_flip=0, initialization="maj")
                maj_acc[index] += (1 - lib.vec_norm(Z[:m] - maj_z, m) / m)
                em_acc[index] += (1 - lib.vec_norm(Z[:m] - em_z, m) / m)
                if run_em0:
                    em0_acc[index] += (1 - lib.vec_norm(Z[:m] - em_z0, m) / m)

                if plot_local:
                    for k in range(K):
                        loc_acc[index, k] += (1 - lib.vec_norm(Z[:m].T - Y[k, :m], m) / m)

            except Exception:
                print("Warning! Convex Solver Failed!")
                error_occ[index] = 1

        for m in range(1, M + 1, delta):
            index = int((m - 1) / delta)
            if error_occ[index] == 1:
                n_vec[index] -= 1

        m_vector = np.linspace(1, M, l)
        fig = plt.figure()
        plt.plot(m_vector, maj_acc / n_vec, 'b', label='maj')
        plt.plot(m_vector, em_acc / n_vec, '--r', label='EM')
        if run_em0:
            plt.plot(m_vector, em0_acc / n_vec, '-.g', label='EM0')
        if plot_local:
            for k in range(K):
                plt.plot(m_vector, (loc_acc[:, k] / n_vec.T).T, label='local' + str(k))

        plt.legend()
        plt.title("Classification Accuracy for $ K = " + str(K) + "$ Local Models and $ N = " + str(n) + "$ Iterations ")
        plt.xlabel("$m$")
        plt.ylabel("Accuracy")
        plt.show()
        fig.savefig('resultx.eps')


