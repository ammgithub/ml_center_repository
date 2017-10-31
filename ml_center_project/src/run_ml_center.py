"""
Created on October 25, 2017

Running ml_center

"""

import numpy as np
from ml_center_module import FastKernelClassifier
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from time import time

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

    print 80 * "-"
    print 18 * " " + "Machine learning center tests (2-class separation)"
    print 80 * "-"
    print 2 * " " + "(1) Testing OR problem"
    print 2 * " " + "(2) Testing AND problem"
    print 2 * " " + "(3) Testing 2-dimensional circular data"
    print 2 * " " + "(4) IRIS dataset: Testing 2-attribute, 2-class version (samples 0,...,99, classes 0, 1)"
    print 2 * " " + "(5) IRIS dataset: Testing 4-attribute, 2-class version (samples 0,...,99, classes 0, 1)"
    print 2 * " " + "(6) BREAST CANCER dataset Testing (all samples)"
    print 2 * " " + "(7) Computing generalization error for BREAST CANCER dataset (100 experiments)"
    print 80 * "-"

    bad_input = True
    while bad_input:
        try:
            user_in = int(raw_input("Make selection: "))
            bad_input = False
        except ValueError as e:
            print "%s is not a valid selection. Please try again. "\
                  % e.args[0].split(':')[1]

    if user_in == 1:
        print "(1) Testing OR problem \n"
        # Testing OR
        trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        trY = [1, -1, 1, -1]
        tsX = np.array([[1, 2], [-3, 2], [6, -1]])
        tsY = [1, -1, 1]

        kernel = 'rbf'; degree = 2; gamma = 1; coef0 = 1
        print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f"\
              % (kernel, degree, gamma, coef0)
        print "-----------------------------------------------------"
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
        fkc.fit(trX, trY)
        print "(fkc.weight_opt, fkc.eps_opt) = \n", (fkc.weight_opt, fkc.eps_opt)
        ftest = fkc.predict(tsX)
        print "tsX = \n", tsX
        print "fkc.predict(tsX) = \n", ftest
        print "tsY = \n", tsY
        if not (abs(ftest - tsY) <= 0.001).all():
            print "*** Test set not classified correctly. ***"
        ftest = fkc.predict(trX)
        print "trX = \n", trX
        print "fkc.predict(trX) = \n", ftest
        print "trY = \n", trY
        if not (abs(ftest - trY) <= 0.001).all():
            print "*** TRAINING SET NOT CLASSIFIED CORRECTLY. ***"
        fkc.plot2d(0.02)
    elif user_in == 2:
        print "(2) Testing AND problem \n"
        # Testing AND
        trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        trY = [1, -1, -1, -1]
        tsX = np.array([[1, 2], [-3, 2], [6, -1]])
        tsY = [1, -1, 1]

        kernel = 'rbf'; degree = 1; gamma = 1; coef0 = 1
        print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f"\
              % (kernel, degree, gamma, coef0)
        print "-----------------------------------------------------"
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
        fkc.fit(trX, trY)
        print "(fkc.weight_opt, fkc.eps_opt) = ", (fkc.weight_opt, fkc.eps_opt)
        ftest = fkc.predict(tsX)
        print "tsX = \n", tsX
        print "fkc.predict(tsX) = \n", ftest
        print "tsY = \n", tsY
        if not (abs(ftest - tsY) <= 0.001).all():
            print "*** Test set not classified correctly. ***"
        ftest = fkc.predict(trX)
        print "trX = \n", trX
        print "fkc.predict(trX) = \n", ftest
        print "trY = \n", trY
        if not (abs(ftest - trY) <= 0.001).all():
            print "*** TRAINING SET NOT CLASSIFIED CORRECTLY. ***"
        fkc.plot2d(0.02)
    elif user_in == 3:
        print "(3) Testing 2-dimensional circular data \n"
        # Testing CIRCLE
        trX = np.array([[1, 1], [4, 1], [1, 4], [4, 4], [2, 2], [2, 3], [3, 2]])
        trY = [1, 1, 1, 1, -1, -1, -1]
        tsX = np.array([[0, 2], [3, 3], [6, 3]])
        tsY = [1, -1, 1]
        kernel = 'rbf'; degree = 2; gamma = 1; coef0 = 1

        print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f" \
              % (kernel, degree, gamma, coef0)
        print "-----------------------------------------------------"
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
        fkc.fit(trX, trY)
        print "(fkc.weight_opt, fkc.eps_opt) = ", (fkc.weight_opt, fkc.eps_opt)
        ftest = fkc.predict(tsX)
        print "tsX = \n", tsX
        print "fkc.predict(tsX) = \n", ftest
        print "tsY = \n", tsY
        if not (abs(ftest - tsY) <= 0.001).all():
            print "*** Test set not classified correctly. ***"
        ftest = fkc.predict(trX)
        print "trX = \n", trX
        print "fkc.predict(trX) = \n", ftest
        print "trY = \n", trY
        if not (abs(ftest - trY) <= 0.001).all():
            print "*** TRAINING SET NOT CLASSIFIED CORRECTLY. ***"
        fkc.plot2d(0.02)
    elif user_in == 4:
        print "(4) Testing 2 attribute, 2-class version of IRIS dataset (samples 0,...,99, classes 0, 1) \n"
        # Testing 2 attribute IRIS
        iris = datasets.load_iris()
        trX = iris.data[:100, :]
        # Classes 0 and 1 only
        trX = trX[:, [0, 1]]
        trY = iris.target[:100]
        trY = [i if i == 1 else -1 for i in trY]
        kernel = 'linear'; degree = 1; gamma = 1; coef0 = 1

        print "\nkernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f"\
              % (kernel, degree, gamma, coef0)
        print "-----------------------------------------------------"
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
        fkc.fit(trX, trY)
        print "(fkc.weight_opt, fkc.eps_opt) = ", (fkc.weight_opt, fkc.eps_opt)
        ftest = fkc.predict(trX)
        print "trX = ", trX
        print "fkc.predict(trX) = ", ftest
        print "trY = ", trY
        if not (abs(ftest - trY) <= 0.001).all():
            print "*** Training set not classified correctly. ***"
        print "No prodiction done."
        fkc.plot2d(0.02)
    elif user_in == 5:
        print "(5) Testing 4 attribute, 2-class version of IRIS dataset (samples 0,...,99, classes 0, 1) \n"
        # Testing 4 attribute IRIS
        iris = datasets.load_iris()
        trX = scaler.fit_transform(iris.data[:100, :])
        # trX = iris.data[:100, :]
        trY = iris.target[:100]
        trY = [i if i == 1 else -1 for i in trY]
        kernel = 'linear'; degree = 1; gamma = 1; coef0 = 1

        print "\nkernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f"\
              % (kernel, degree, gamma, coef0)
        print "-----------------------------------------------------"
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
        fkc.fit(trX, trY)
        print "(fkc.weight_opt, fkc.eps_opt) = ", (fkc.weight_opt, fkc.eps_opt)
        ftest = fkc.predict(trX)
        print "trX = ", trX
        print "fkc.predict(trX) = ", ftest
        print "trY = ", np.array(trY, 'float')
        if not (abs(ftest - trY) <= 0.001).all():
            print "*** Training set not classified correctly. ***"
        print "No prodiction done."
        fkc.plot2d(0.02)
    elif user_in == 6:
        print "(6) Testing BREAST CANCER dataset (all samples) \n"
        #####################################################################
        # Testing BREAST CANCER                                             #
        # Classes:              2                                           #
        # Samples per class:    212(Malignant), 357(Benign)                 #
        # Samples total:        569                                         #
        # Dimensionality:       30                                          #
        # Features:             real, positive                              #
        #####################################################################
        bc_data = datasets.load_breast_cancer()
        scaler = MinMaxScaler()
        trX = bc_data.data
        trX = scaler.fit_transform(trX)
        trY = bc_data.target
        trY = np.array([i if i == 1 else -1 for i in trY])
        kernel = 'rbf'; degree = 1; gamma = 8; coef0 = 1

        print "\nkernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f"\
              % (kernel, degree, gamma, coef0)
        print "-----------------------------------------------------"
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
        fkc.fit(trX, trY)
        print "(fkc.weight_opt, fkc.eps_opt) = ", (fkc.weight_opt, fkc.eps_opt)
        ftest = fkc.predict(trX)
        print "Skipped printing trX ...\n "
        print "fkc.predict(trX) = \n", ftest
        print "trY = \n", trY
        if not (abs(ftest - trY) <= 0.001).all():
            print "*** TRAINING SET NOT CLASSIFIED CORRECTLY. ***"
        print "No prodiction done."
        fkc.plot2d(0.02)
    elif user_in == 7:
        print "Computing generalization error for BREAST CANCER dataset (100 experiments)\n"
        #####################################################################
        # Testing BREAST CANCER                                             #
        # Classes:              2                                           #
        # Samples per class:    212(Malignant), 357(Benign)                 #
        # Samples total:        569                                         #
        # Dimensionality:       30                                          #
        # Features:             real, positive                              #
        #####################################################################

        # All inputs and all labels
        bc_data = datasets.load_breast_cancer()
        scaler = MinMaxScaler()
        trx_all = bc_data.data
        trx_all = scaler.fit_transform(trx_all)
        try_all = bc_data.target
        try_all = np.array([i if i == 1 else -1 for i in try_all])
        (num_samples, num_features) = trx_all.shape

        # Training-to-test ratio of 67% : 33% (Bennett, Mangasarian, 1992)
        train_ratio = 0.67
        test_ratio = 0.33
        num_train_samples = int(round(.67*num_samples, 0))  # 381 train
        num_test_samples = int(round(.33*num_samples, 0))  # 188 test
        assert num_train_samples + num_test_samples == num_samples, \
            "Please check the number of training and test samples. "

        t = time()
        gen_error_list = []
        np.random.seed(0)
        num_sims = 100
        for i in range(10):
            # sorting not necessary, np.array not necessary
            tr_idx = np.random.choice(num_samples, num_train_samples,
                                             replace=False)
            ts_idx = list(set(range(num_samples)) - set(list(tr_idx)))
            trX = trx_all[tr_idx, :]
            trY = try_all[tr_idx]
            tsX = trx_all[ts_idx, :]
            tsY = try_all[ts_idx]

            kernel = 'rbf';degree = 1;gamma = 10;coef0 = 1
            fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
            fkc.fit(trX, trY)
            ftest = fkc.predict(tsX)
            num_wrong = 1 * (ftest != tsY).sum()
            gen_error_list.append(num_wrong / float(num_test_samples))

        print "Generatlization error BREAST CANCER dataset (100 experiments): \n", \
            np.array(gen_error_list)
        print "Elapsed time %4.1f seconds." % (time() - t)
    else:
        print "Invalid selection. Program terminating. "
    print "Finished."




