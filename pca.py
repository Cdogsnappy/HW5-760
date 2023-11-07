import numpy as np



#Demeaned and Normalized are metaparameters, ONLY ONE SHOULD BE SET TO TRUE AT A TIME.
def pca(components, hyper_comp, demeaned=False, normalized=False):
    mean = np.mean(components, axis=0)
    st_norm = 1/np.std(components, axis=0)
    if(demeaned):
        components -= mean
    if(normalized):
        components-=mean
        components*=st_norm

    cov = np.cov(components.T)
    eig_vals, eig_vecs = np.linalg.eig(cov)

    threshold = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[threshold]
    eig_vecs = eig_vecs[:, threshold]

    trans_comp = eig_vecs[:hyper_comp]
    var = np.sum(eig_vals[:hyper_comp]) / np.sum(eig_vals)
    print("explained variance: " + str(var))

    return np.dot(components, trans_comp.T)


X_old = np.random.normal(loc=0, scale=1, size=(1000, 10))
X_new = np.random.normal(loc=0, scale=1, size=(500, 10))

print(pca(X_old, 8, normalized=True).shape)
