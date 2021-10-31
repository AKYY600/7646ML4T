import numpy as np
import DTLearner as dt
import RTLearner as rt
import LinRegLearner as lrl


class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
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

    def query(self, points):
        Y_test = []
        for m in range(self.bags):
            each_Y = self.learners[m].query(points)
            Y_test.append(each_Y)
        Y_final = np.mean(Y_test, axis = 0)
        return Y_final

