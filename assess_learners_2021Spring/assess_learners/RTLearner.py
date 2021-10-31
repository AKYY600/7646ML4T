import numpy as np
from scipy.stats import mode

class RTLearner(object):
    def __init__(self, leaf_size, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return "yliu3306"

    def find_feature(self,feature,y):
        index = np.random.randint(feature.shape[1])
        return index

    def add_evidence(self, data_x, data_y):
        if data_x.shape[0]==1:
            self.tree = np.array([[-1,data_y,-1,-1]], dtype=float)
            return self.tree
        if np.isclose(data_y,data_y[0]).all():
            self.tree = np.array([[-1,data_y[0],-1,-1]], dtype=float)
            return self.tree
        if data_x.shape[0] <= self.leaf_size:
            self.tree = np.array([[-1,mode(data_y)[0][0],-1,-1]], dtype=float)
            #self.tree = np.array([[-1, np.mean(data_y), -1, -1]], dtype=float)
            return self.tree

        else:
            i = self.find_feature(data_x,data_y)
            p1 = np.random.randint(data_x.shape[0])
            p2 = np.random.randint(data_x.shape[0])
            SplitVal = (data_x[p1,i]+data_x[p2,i])/2

            if SplitVal >= np.max(data_x[:,i]):
                self.tree = np.array([[-1, np.mean(data_y), -1, -1]], dtype=float)
                return self.tree
            lefttree = np.array(self.add_evidence(data_x[data_x[:, i] <= SplitVal], data_y[data_x[:, i] <= SplitVal]))
            righttree = np.array(self.add_evidence(data_x[data_x[:, i] > SplitVal], data_y[data_x[:, i] > SplitVal]))

            root = np.array([[i, SplitVal, 1, lefttree.shape[0] + 1]], dtype=float)

            self.tree = np.vstack((root, lefttree))
            self.tree = np.vstack((self.tree, righttree))
            return self.tree

    def query(self, points):
        result = []
        tree = np.array(self.tree)

        for i in range(0, points.shape[0]):
            j = 0
            while self.tree[j, 0] != -1:
                splitV = tree[j, 1]
                index = int(tree[j, 0])

                if points[i, index] <= splitV:
                    j = j + 1

                else:
                    j = j + int(tree[j, 3])

            result.append(float(tree[j, 1]))

        return result

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
