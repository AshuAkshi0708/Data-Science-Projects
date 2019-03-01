#!/usr/bin/env python

import numpy as np
from sklearn.decomposition import PCA

from config import *

if __name__ == "__main__":
    r = np.load("representation-" + str(C_SIZE) + ".npy");
    pca = PCA(n_components = U_SIZE);
    cr = pca.fit_transform(r);
    np.save("representation-" + str(U_SIZE) + ".npy", cr);
