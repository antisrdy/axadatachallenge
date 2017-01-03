import numpy as np

n_burn_in = 672


class FeatureExtractor(object):

    def __init__(self):
        pass

    def transform(self, X_ds):
        temp = X_ds.values.reshape((-1,1)).copy()
        result = temp.copy()
        for i in range(1, 672):
            result = np.hstack((result, np.roll(temp, i)))
        return result[n_burn_in::, :]