""""""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Atlanta, Georgia 30332  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
All Rights Reserved  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Template code for CS 4646/7646  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
or edited.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT honor code violation.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
-----do not edit anything above this line---  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Student Name: Tucker Balch (replace with your name)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT User ID: tb34 (replace with your User ID)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT ID: 900897987 (replace with your GT ID)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import warnings  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import numpy as np  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
class DTLearner(object):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    This is a decision tree learner object that is implemented incorrectly. You should replace this DTLearner with  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    your own correct DTLearner from Project 3.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param leaf_size: The maximum number of samples to be aggregated at a leaf, defaults to 1.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type leaf_size: int  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type verbose: bool  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    def __init__(self, leaf_size = 1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return "yliu3306"

    def find_feature(self,feature,y):
        max = 0
        index = 0
        for n in range(0,feature.shape[1]):
            if np.correlate(feature[:,n],y)>max:
                max = np.correlate(feature[:,n],y)
                index = n
        return index

    def add_evidence(self, data_x, data_y):
        if data_x.shape[0]==1:
            self.tree = np.array([[-1,data_y,-1,-1]], dtype=float)
            return self.tree
        if np.isclose(data_y,data_y[0]).all():
            self.tree = np.array([[-1,data_y[0],-1,-1]], dtype=float)
            return self.tree
        if data_x.shape[0] <= self.leaf_size:
            self.tree = np.array([[-1,np.mean(data_y),-1,-1]], dtype=float)
            return self.tree

        else:
            i = self.find_feature(data_x,data_y)
            SplitVal = np.median(data_x[:,i], axis=0)

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

