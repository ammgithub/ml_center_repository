"""
Created on October 25, 2017

Running ml_center

"""

import numpy as np
from ml_center_module import FastKernelClassifier
from sklearn import datasets

__author__ = 'amm'
__date__ = "Oct 25, 2017"
__version__ = 0.0

np.set_printoptions(linewidth=100, edgeitems='all', suppress=True,
                    precision=4)

if __name__ == '__main__':
    """
    execfile('run_ml_center.py')
    """
    import os
    os.chdir('C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\ml_center_project\\src')

    # Testing OR data
    # print "Testing OR:"
    # trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    # trY = [1, -1, 1, -1]
    # tsX = np.array([[1, 2], [-3, 2], [6, -1]])
    # tsY = [1, -1, 1]
    # Testing AND data
    # print "Testing AND:"
    # trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    # trY = [1, -1, -1, -1]
    # tsX = np.array([[1, 2], [-3, 2], [6, -1]])
    # tsY = [1, -1, 1]
    # Testing CIRCLE data
    # print "\nTesting CIRCLE:"
    # trX = np.array([[1, 1], [4, 1], [1, 4], [4, 4], [2, 2], [2, 3], [3, 2]])
    # trY = [1, 1, 1, 1, -1, -1, -1]
    # tsX = np.array([[0, 2], [3, 3], [6, 3]])
    # tsY = [1, -1, 1]

    # Running OR, AND, and CIRCLE
    # kernel = 'rbf'
    # degree = 2
    # gamma = 1
    # coef0 = 1
    # print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f"%(kernel, degree, gamma, coef0)
    # print "-----------------------------------------------------"
    #
    # fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
    # fkc.fit(trX, trY)
    # print "(fkc.weight_opt, fkc.eps_opt) = ", (fkc.weight_opt, fkc.eps_opt)
    # ftest = fkc.predict(tsX)
    # print "fkc.predict(tsX) = \n", ftest
    # print "tsY = \n", tsY
    # if not (abs(ftest - tsY) <= 0.001).all():
    #     print "*** Test set not classified correctly. ***"
    # ftest = fkc.predict(trX)
    # print "fkc.predict(trX) = \n", ftest
    # print "trY = \n", trY
    # if not (abs(ftest - trY) <= 0.001).all():
    #     print "*** TRAINING SET NOT CLASSIFIED CORRECTLY. ***"
    # fkc.plot2d(0.02)

    # Running IRIS
    # iris = datasets.load_iris()
    # # trX = iris.data[:100, [0, 1]]
    # trX = iris.data[:100, :]
    # trY = iris.target[:100]
    # trY = [i if i==1 else -1 for i in trY]
    #
    # kernel = 'linear'
    # degree = 1
    # gamma = 1
    # coef0 = 1
    # print "\nkernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f"%(kernel, degree, gamma, coef0)
    # print "-----------------------------------------------------"
    #
    # fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
    # fkc.fit(trX, trY)
    # print "(fkc.weight_opt, fkc.eps_opt) = ", (fkc.weight_opt, fkc.eps_opt)
    # ftest = fkc.predict(trX)
    # print "fkc.predict(trX) = ", ftest
    # print "trY = ", trY
    # if not (abs(ftest - trY) <= 0.001).all():
    #     print "*** Training set not classified correctly. ***"
    # fkc.plot2d(0.02)

    # Running BREAST CANCER
    bc_data = datasets.load_breast_cancer()
    trX = bc_data.data
    trY = bc_data.target
    trY = np.array([i if i == 1 else -1 for i in trY])

    kernel = 'rbf'
    degree = 4
    gamma = 1.0
    coef0 = 1
    print "\nkernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f"%(kernel, degree, gamma, coef0)
    print "-----------------------------------------------------"

    fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
    fkc.fit(trX, trY)
    print "(fkc.weight_opt, fkc.eps_opt) = ", (fkc.weight_opt, fkc.eps_opt)
    ftest = fkc.predict(trX)
    print "fkc.predict(trX) = \n", ftest
    print "trY = \n", trY
    if not (abs(ftest - trY) <= 0.001).all():
        print "*** TRAINING SET NOT CLASSIFIED CORRECTLY. ***"
    fkc.plot2d(0.02)
