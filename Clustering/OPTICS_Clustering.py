import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN, OPTICS

"""
X,_ = make_blobs(200, 2, centers=4, cluster_std=1.3)

Xs = ((X- np.mean(X, axis=0))/np.std(X,axis=0))

oc = OPTICS()
oc.fit(Xs)

labels_unique = np.unique(oc.labels_)
n_clusters = len(labels_unique)
print(n_clusters)

out = np.stack([oc.labels_, oc.reachability_,oc.reachability_[oc.ordering_], oc.ordering_, oc.core_distances_])
out = np.hstack([Xs,out.T])

np.savetxt('optics_out.txt', out)
"""


"""
X = np.loadtxt('dataset.txt', skiprows = 1, usecols = (1,2,3,4,5,6))

Xs = ((X- np.mean(X, axis=0))/np.std(X,axis=0))

oc = OPTICS()
oc.fit(Xs)

labels_unique = np.unique(oc.labels_)
n_clusters = len(labels_unique)
print(n_clusters)

out = np.stack([oc.labels_, oc.reachability_,oc.reachability_[oc.ordering_], oc.ordering_, oc.core_distances_])
out = np.hstack([Xs,out.T])

np.savetxt('optics_data.txt', out)
"""

X = np.loadtxt('cumulants.txt')

Xs = ((X- np.mean(X, axis=0))/np.std(X,axis=0))

print(X)

oc = OPTICS()
oc.fit(Xs)

labels_unique = np.unique(oc.labels_)
n_clusters = len(labels_unique)
print(n_clusters)

out = np.stack([oc.labels_, oc.reachability_,oc.reachability_[oc.ordering_], oc.ordering_, oc.core_distances_])
out = np.hstack([X,out.T])

np.savetxt('optics_cumulants.txt', out)


