import numpy as np

# Piratiado: https://kaushikghose.wordpress.com/2013/10/24/computing-mutual-information-and-other-scary-things/

def calc_MI(X, Y, bins=100, sparse=False):
    if sparse:
        X = X.toarray()[0]
        Y = Y.toarray()[0]

    c_XY = np.histogram2d(X,Y,bins)[0]
    c_X = np.histogram(X,bins)[0]
    c_Y = np.histogram(Y,bins)[0]

    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    return MI

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H


