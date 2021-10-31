""""""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Test a learner.  (c) 2015 Tucker Balch  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
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
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import math  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import sys
import time
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import numpy as np
import matplotlib.pyplot as plt
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
if __name__ == "__main__":  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    if len(sys.argv) != 2:  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        print("Usage: python testlearner.py <filename>")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        sys.exit(1)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    inf = open(sys.argv[1])
    print (inf)
    print (type(inf))
    data = np.array(
        [list(map(float, s.strip().split(",")[1:])) for s in inf.readlines()[1:]]
    )  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # compute how much of the data is training and testing  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    train_rows = int(0.6 * data.shape[0])  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    test_rows = data.shape[0] - train_rows  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # separate out training and testing data  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    train_x = data[:train_rows, 0:-1]  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    train_y = data[:train_rows, -1]  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    test_x = data[train_rows:, 0:-1]  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    test_y = data[train_rows:, -1]  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"{test_x.shape}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"{test_y.shape}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # create a learner and train it  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    #learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner

    #experiment 1
    plotDT = np.array([0,0,0])

    for i in range (1, train_x.shape[0]):
        learner = dt.DTLearner(leaf_size = i, verbose = True)
        learner.add_evidence(train_x, train_y)  # train it


    # evaluate in sample  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        pred_y = learner.query(train_x)  # get the predictions
        rmse_train = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        print()
        print("In sample results")
        print(f"RMSE: {rmse_train}")
        c_train = np.corrcoef(pred_y, y=train_y)
        print(f"corr: {c_train[0,1]}")
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # evaluate out of sample  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        pred_y = learner.query(test_x)  # get the predictions
        rmse_test = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        print("Out of sample results")
        print(f"RMSE: {rmse_test}")
        c_test = np.corrcoef(pred_y, y=test_y)
        print(f"corr: {c_test[0,1]}")

        plotDT = np.vstack([plotDT, np.array([i,rmse_train,rmse_test])])


    plt.figure()
    plt.xlabel('Leaf size')
    plt.ylabel('RMSE')
    plt.plot(plotDT[1:,1], label = "Training set")
    plt.plot(plotDT[1:,2], label = "Test set")
    plt.legend(loc='lower right')
    plt.savefig('Figure1.png')

    plt.figure()
    plt.xlabel('Leaf size')
    plt.ylabel('RMSE')
    plt.plot(plotDT[1:30,1], label = "Training set")
    plt.plot(plotDT[1:30,2], label = "Test set")
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('Figure3.png')

#experiment 2
    plotBL = np.array([0, 0, 0])

    for i in range(1, train_x.shape[0]):
        learner = bl.BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size":i}, bags = 20, boost = False, verbose = False)
        learner.add_evidence(train_x, train_y)  # train it
        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        rmse_train = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        #print("In sample results")
        #print(f"RMSE: {rmse_train}")
        c_train = np.corrcoef(pred_y, y=train_y)
        #print(f"corr: {c_train[0, 1]}")

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        rmse_test = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        #print()
        #print("Out of sample results")
        #print(f"RMSE: {rmse_test}")
        c_test = np.corrcoef(pred_y, y=test_y)
        #print(f"corr: {c_test[0, 1]}")

        plotBL = np.vstack([plotBL, np.array([i, rmse_train, rmse_test])])

    plt.figure()
    plt.xlabel('Leaf size')
    plt.ylabel('RMSE')
    plt.plot(plotBL[1:, 1], label="Training set")
    plt.plot(plotBL[1:, 2], label="Test set")
    plt.legend(loc='lower right')
    plt.savefig('Figure2.png')

    plt.figure()
    plt.xlabel('Leaf size')
    plt.ylabel('RMSE')
    plt.plot(plotBL[1:30, 1], label="Training set")
    plt.plot(plotBL[1:30, 2], label="Test set")
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('Figure4.png')

#experiment 3

    plot_mae_train = np.array([0, 0, 0])
    plot_mae_test = np.array([0, 0, 0])
    plot_time = np.array([0, 0, 0, 0, 0])

    for i in range(1, train_x.shape[0]):
        #DT
        DTlearner = dt.DTLearner(leaf_size=i, verbose=False)
        DTleanerTime = time.time()
        DTlearner.add_evidence(train_x, train_y)  # train it
        DTlearnerTime = time.time() - DTleanerTime

        # evaluate in sample
        DTqueryTime = time.time()
        pred_y1 = DTlearner.query(train_x)  # get the predictions
        mae_dt_train = sum(np.abs(train_y - pred_y1)) / train_y.shape[0]

        DTqueryTime = time.time()-DTqueryTime
        # evaluate out of sample
        pred_y2 = DTlearner.query(test_x)  # get the predictions
        mae_dt_test = sum(np.abs(test_y - pred_y2)) / test_y.shape[0]

        # RT
        RTlearner = rt.RTLearner(leaf_size=i, verbose=False)
        RTleanerTime = time.time()
        RTlearner.add_evidence(train_x, train_y)  # train it
        RTlearnerTime = time.time() - RTleanerTime

        # evaluate in sample
        RTqueryTime = time.time()
        pred_y3 = RTlearner.query(train_x)  # get the predictions
        mae_rt_train = sum(np.abs(train_y - pred_y3)) / train_y.shape[0]
        RTqueryTime = time.time() - RTqueryTime

        # evaluate out of sample
        pred_y4 = RTlearner.query(test_x)  # get the predictions
        mae_rt_test = sum(np.abs(test_y - pred_y4)) / test_y.shape[0]

        plot_mae_train = np.vstack([plot_mae_train, np.array([i, mae_dt_train, mae_rt_train])])
        plot_mae_test = np.vstack([plot_mae_test, np.array([i, mae_dt_test, mae_rt_test])])
        plot_time = np.vstack([plot_time,np.array([i,DTlearnerTime,DTqueryTime,RTlearnerTime,RTqueryTime])])


    plot_time = np.delete(plot_time,(0),axis=0)
    plot_mae_train = np.delete(plot_mae_train,(0),axis=0)
    plot_mae_test = np.delete(plot_mae_test,(0),axis =0)

#plot for time
    plt.figure()
    plt.xlabel('Leaf size')
    plt.ylabel('Time')
    plt.plot(plot_time[:50, 1], label="DT Training time")
    plt.plot(plot_time[:50, 3], label="RT Training time")
    plt.legend(loc='lower right')
    plt.savefig('Figure5.png')

    plt.figure()
    plt.xlabel('Leaf size')
    plt.ylabel('Time')
    plt.plot(plot_time[:50, 2], label="DT Test time")
    plt.plot(plot_time[:50, 4], label="RT Test time")
    plt.legend(loc='lower right')
    plt.savefig('Figure6.png')

#plot for Var
    plt.figure()
    plt.xlabel('Leaf size')
    plt.ylabel('MAE')
    plt.plot(plot_mae_train[:50, 1], label="DT MAE")
    plt.plot(plot_mae_train[:50, 2], label="RT MAE")
    plt.legend(loc='lower right')
    plt.savefig('Figure7.png')

    plt.figure()
    plt.xlabel('Leaf size')
    plt.ylabel('MAE')
    plt.plot(plot_mae_test[:50, 1], label="DT MAE")
    plt.plot(plot_mae_test[:50, 2], label="RT MAE")
    plt.legend(loc='lower right')
    plt.savefig('Figure8.png')