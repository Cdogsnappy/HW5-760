import numpy as np
import matplotlib.pyplot as plt


# Demeaned and Normalized are metaparameters, ONLY ONE SHOULD BE SET TO TRUE AT A TIME.
def pca(components, hyper_comp, demeaned=False, normalized=False):
    mean = np.mean(components, axis=0)
    st_norm = 1 / np.std(components, axis=0)
    if demeaned:
        components -= mean
    if normalized:
        components -= mean
        components *= st_norm

    S, K, U = np.linalg.svd(components)
    trans_comp = U[:hyper_comp]

    return np.dot(components, trans_comp.T), trans_comp, mean, st_norm


def buildData(f):
    X = []
    f.readline()
    for line in f:
        l = line.split(',')
        list1 = np.array([float(number) for number in l[0:len(l)]])
        X.append(list1)
    return X


def dro(components, hyper_comp):
    mean = np.mean(components, axis=0)
    components -= mean
    S, K, U = np.linalg.svd(components)
    trans_comp = U[:hyper_comp]
    return np.dot(components, trans_comp.T), trans_comp.T, mean


f1 = open('data2D.csv', 'r')
X_1 = np.array(buildData(f1))
# PCAS
buggy = pca(np.copy(X_1), 1)
demeaned = pca(np.copy(X_1), 1, demeaned=True)
normalized = pca(np.copy(X_1), 1, normalized=True)
Z, A, b = dro(np.copy(X_1), 1)

# GRAPHS
plt.scatter(X_1[:, 0], X_1[:, 1], c='green')
recon_buggy = np.dot(buggy[0], buggy[1])
plt.scatter(recon_buggy[:, 0], recon_buggy[:, 1], c='red')
plt.title("Buggy")
plt.show()
plt.scatter(X_1[:, 0], X_1[:, 1], c='green')
recon_demeaned = np.dot(demeaned[0], demeaned[1]) + demeaned[2]
plt.scatter(recon_demeaned[:, 0], recon_demeaned[:, 1], c='red')
plt.title("Demeaned")
plt.show()
plt.scatter(X_1[:, 0], X_1[:, 1], c='green')
recon_normed = np.dot(normalized[0], normalized[1]) * (1 / normalized[3]) + normalized[2]
plt.scatter(recon_normed[:, 0], recon_normed[:, 1], c='red')
plt.title("Normalized")
plt.show()
plt.scatter(X_1[:, 0], X_1[:, 1], c='green')
recon_dro = np.dot(Z, A.T) + b
plt.scatter(recon_dro[:, 0], recon_dro[:, 1], c='red')
plt.title("DRO")
plt.show()
print("BUGGY ERROR 2D = " + str(np.sum(np.square(np.linalg.norm(X_1 - recon_buggy)))))
print("DEMEANED ERROR 2D = " + str(np.sum(np.square(np.linalg.norm(X_1 - recon_demeaned)))))
print("NORMALIZED ERROR 2D = " + str(np.sum(np.square(np.linalg.norm(X_1 - recon_normed)))))
print("DRO ERROR 2D = " + str(np.sum(np.square(np.linalg.norm(X_1 - recon_dro)))))
print()

# 1000D DATASET
f2 = open('data1000D.csv', 'r')
X_1000 = np.array(buildData(f2))
buggy_1000 = pca(np.copy(X_1000), 498)
demeaned_1000 = pca(np.copy(X_1000), 498, demeaned=True)
normalized_1000 = pca(np.copy(X_1000), 498, normalized=True)
Z_1000, A_1000, b_1000 = dro(np.copy(X_1000), 498)
recon_buggy_1000 = np.dot(buggy_1000[0], buggy_1000[1])
recon_demeaned_1000 = np.dot(demeaned_1000[0], demeaned_1000[1]) + demeaned_1000[2]
recon_normed_1000 = np.dot(normalized_1000[0], normalized_1000[1]) * (1 / normalized_1000[3]) + normalized_1000[2]
recon_dro_1000 = np.dot(Z_1000, A_1000.T) + b_1000
print("BUGGY ERROR 1000D = " + str(np.sum(np.square(np.linalg.norm(X_1000 - recon_buggy_1000)))))
print("DEMEANED ERROR 1000D = " + str(np.sum(np.square(np.linalg.norm(X_1000 - recon_demeaned_1000)))))
print("NORMALIZED ERROR 1000D = " + str(np.sum(np.square(np.linalg.norm(X_1000 - recon_normed_1000)))))
print("DRO ERROR 1000D = " + str(np.sum(np.square(np.linalg.norm(X_1000 - recon_dro_1000)))))
