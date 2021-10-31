import numpy as np
import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.learners = []
        self.bbags = 20
        for i in range (0,self.bbags):
            self.learners.append(bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = 20, boost = False, verbose = False))
    def author(self):
        return "yliu3306"  # replace tb34 with your Georgia Tech username
    def add_evidence(self, data_x, data_y):
        for m in range(self.bbags):
            self.learners[m].add_evidence(data_x, data_y)
    def query(self, points):
        Y_test = []
        for m in range(self.bbags):
            each_Y = self.learners[m].query(points)
            Y_test.append(each_Y)
        return np.mean(Y_test, axis=0)

