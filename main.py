import math
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

iter = 100
P_a = ([-1, -1], [[1, .5], [.5, 1]])
P_b = ([1, -1], [[1, -.5], [-.5, 2]])
P_c = ([0, 1], [[1, 0], [0, 2]])
reg_term = 1e-6 * np.identity(2)


def gmm(X):
    means = np.random.randint(min(X[:, 0]), max(X[:, 0]), size=(3, len(X[0])))
    cov = np.zeros((3, 2, 2))
    for dim in range(len(cov)):
        np.fill_diagonal(cov[dim], 5)
    pi = np.ones(3) / 3

    r_ic = np.zeros((len(X), len(cov)))
    for val in range(len(r_ic[0])):
        mn = multivariate_normal(mean=means[val], cov=cov[val])
        denom = np.zeros(len(X))
        for j in range(len(means)):
            denom[j] = pi[j] * multivariate_normal(mean=means[j], cov=cov[j]).pdf(X)
        r_ic[:, val] = pi[val] * mn.pdf(X) / np.sum(denom, axis=0)

    means = []
    cov = []
    pi = []
    for val in range(len(r_ic[0])):
        mu_val = np.sum(r_ic[:, val], axis=0)
        new_mean = (1 / mu_val) * np.sum(X * r_ic[:, val].reshape(len(X), 1), axis=0)
        means.append(new_mean)
        cov.append(((1 / mu_val) * np.dot((np.array(r_ic[:, c]).reshape(len(X), 1) * (X - new_mean)).T,
                                          (X - new_mean))) + reg_term)
        pi.append(mu_val / np.sum(r_ic))


    # generate the dataset


def k_means(X, num_clusters):
    clusters = smart_cluster(X, num_clusters)
    cluster_classes = []
    prev_loss = 0
    loss = -1
    while not math.isclose(prev_loss, loss):
        cluster_classes = []
        for c in clusters:
            cluster_classes.append([])
        for i in range(num_clusters - 1):
            dist = np.zeros((len(clusters), len(X)))
            for j in range(len(clusters)):
                dist[j,:] = np.linalg.norm(X - clusters[j], axis=1)
            for x, y in zip(X, np.argmin(dist, axis=0)):
                cluster_classes[y].append(x)
        for val in range(len(clusters)):
            clusters[val] = np.sum(cluster_classes[val], axis=0) / len(cluster_classes[val])
        prev_loss = loss
        loss = 0
        for f in range(len(cluster_classes)):
            loss += np.sum(np.square(np.linalg.norm(cluster_classes[f] - clusters[f])))

    return loss, cluster_classes, clusters


def smart_cluster(X, num_clusters):
    clusters = [X[np.random.randint(0, len(X))]]
    D = []
    for i in range(num_clusters - 1):
        dist = np.zeros((len(clusters), len(X)))
        for j in range(len(clusters)):
            dist[j, :] = np.linalg.norm(X - clusters[j], axis=1)
        if len(clusters) == 1:
            D = dist
        else:
            D = [dist[j, i] for i, j in zip(range(len(dist[0])), np.argmin(dist, axis=0))]
        D = np.square(D) / np.sum(np.square(D))
        D = D.flatten()
        next_cluster = X[np.random.choice(range(300), p=D)]
        clusters.append(next_cluster)
    return clusters


def q1_2():
    print()


sig = [.5, 1, 2, 4, 8]
tot_loss = []
acc = []
base_means = [P_a[0], P_b[0], P_c[0]]
for s in sig:

    a = np.random.multivariate_normal(P_a[0], s * np.array(P_a[1]), 100)
    b = np.random.multivariate_normal(P_b[0], s * np.array(P_b[1]), 100)
    c = np.random.multivariate_normal(P_c[0], s * np.array(P_c[1]), 100)
    X = np.append(a, b, 0)
    X = np.append(X, c, 0)
    X = np.array(X)
    loss, classes, clusters, = k_means(X, 3)
    tot_loss.append(loss)
    idx = []
    for g in clusters:
        diff = np.linalg.norm((base_means - g))
        idx.append(np.argmin(diff, axis=0))
    classes[:] = [classes[i] for i in idx]  # Reordered classes
    correct = 0
    for a_i, b_i, c_i in zip(a, b, c):
        if (classes[0] == a_i).any(): correct += 1
        if (classes[1] == b_i).any(): correct += 1
        if (classes[2] == c_i).any(): correct += 1
    acc.append(correct / 300)
plt.plot(sig, tot_loss)
plt.xlabel("sigma")
plt.ylabel("loss")
plt.show()
plt.plot(sig, acc)
plt.xlabel("sigma")
plt.ylabel("accuracy")
plt.show()
