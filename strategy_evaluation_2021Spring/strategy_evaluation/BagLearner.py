import numpy as np

import RTLearner as rt

from scipy.stats import mode


class BagLearner(object):

    def __init__(self, learner=rt.RTLearner, kwargs= {"leaf_size":10, "verbose":False}, bags=20, boost=False, verbose=False):
        self.learners = []
        self.learner = learner
        self.bags = bags
        for i in range(0, self.bags):
            self.learners.append(self.learner(**kwargs))
    def author(self):
        return "yliu3306"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        n = data_x.shape[0]
        for m in range(self.bags):
            c_rows = np.random.choice(data_x.shape[0], size=n, replace=True)
            self.learners[m].add_evidence(data_x[c_rows], data_y[c_rows])
            #self.learners[m].add_evidence(data_x, data_y)

    def query(self, points):
        Y_test = []
        for m in range(self.bags):
            each_Y = self.learners[m].query(points)
            Y_test.append(each_Y)
        Y_final = mode(Y_test)[0][0]
        return Y_final

        '''
        k = points.shape[0]
        Y_test = np.array([0]*k)[np.newaxis]
        Y_final = np.array([])
        for m in range(self.bags):
            each_Y = self.learners[m].query(points)
            each_Y = each_Y[np.newaxis]
            Y_test = np.vstack((Y_test,each_Y))

        Y_test = Y_test[1:,:]

        for n in range(Y_test.shape[1]):
            Y_final = np.append(Y_final, mode(Y_test[:,n])[0][0])

        return Y_final
        '''

