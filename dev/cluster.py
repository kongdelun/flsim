import numpy as np
from k_means_constrained import KMeansConstrained

X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
clf = KMeansConstrained(
    n_clusters=2,
    size_min=2,
    size_max=5,
    random_state=0
)
clf.fit_predict(X)
print(clf.labels_)